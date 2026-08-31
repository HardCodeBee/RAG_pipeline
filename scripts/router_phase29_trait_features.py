"""Feature extraction for the Phase 2.9 single-query-trait audit.

Every value in this module is available before the main BM25/BGE retriever is
executed. Some Stage-C values perform an unranked query-conditioned aggregate
over a fixed, label-independent corpus-sample index; they never expose a main
retriever ranked list, qrels, or generated answers.
"""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.retrievers.sqlite_bm25 import analyze_sqlite_bm25_text


RAW_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
QUOTED_RE = re.compile(r'["“”]([^"“”]+)["“”]')
YEAR_RE = re.compile(r"(?:1[5-9]|20)\d{2}")
CAPITALIZED_RE = re.compile(r"^[A-Z][a-z]+(?:['’][A-Za-z]+)?$")
ACRONYM_RE = re.compile(r"^[A-Z]{2,}(?:\d+)?$")
ALPHANUMERIC_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)")

WH_WORDS = {"who", "whom", "whose", "where", "when", "what", "which", "why", "how"}
QUESTION_AUXILIARIES = {
    "am", "are", "is", "was", "were", "be", "been", "being", "do", "does", "did",
    "has", "have", "had", "can", "could", "will", "would", "shall", "should", "may",
    "might", "must",
}
FUNCTION_WORDS = WH_WORDS | QUESTION_AUXILIARIES | {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "from", "by", "with",
    "and", "or", "but", "as", "than", "that", "this", "these", "those", "it", "its",
    "he", "she", "they", "their", "his", "her", "them", "we", "our", "you", "your",
}
PRONOUNS = {
    "he", "she", "it", "they", "them", "his", "her", "hers", "its", "their", "theirs",
    "this", "that", "these", "those", "who", "whom", "whose", "which",
}
DEMONSTRATIVES = {"this", "that", "these", "those", "former", "latter"}
NEGATIONS = {"not", "never", "no", "neither", "nor", "without"}
TEMPORAL_HINTS = {
    "year", "date", "day", "month", "century", "decade", "before", "after", "during",
    "when", "born", "died", "founded", "formed", "released", "published", "opened",
}
LOCATION_HINTS = {
    "where", "country", "city", "state", "province", "county", "district", "region",
    "island", "continent", "place", "location", "capital", "river", "mountain",
}
PERSON_HINTS = {"who", "whom", "person", "man", "woman", "actor", "actress", "writer", "author"}
COMPARISON_HINTS = {
    "both", "either", "same", "different", "older", "younger", "earlier", "later", "more",
    "less", "longer", "shorter", "higher", "lower", "first", "second", "which",
}
SUPERLATIVE_HINTS = {
    "most", "least", "best", "worst", "largest", "smallest", "highest", "lowest", "oldest",
    "youngest", "earliest", "latest", "longest", "shortest",
}
BRIDGE_HINTS = {
    "whose", "which", "that", "where", "who", "associated", "related", "known", "played",
    "worked", "married", "founded", "directed", "written", "located", "owned", "member",
}
RELATION_HINTS = {
    "born", "died", "founded", "formed", "released", "published", "written", "wrote", "writer",
    "directed", "director", "played", "member", "owned", "operated", "located", "married",
    "served", "represented", "created", "invented", "developed", "produced", "sponsored",
    "won", "awarded", "attended", "graduated", "studied", "worked", "appeared", "starred",
    "based", "adapted", "inspired", "includes", "contains", "features", "named", "known",
}
GENERIC_HEADS = {
    "thing", "things", "person", "people", "place", "places", "group", "company", "organization",
    "event", "work", "name", "type", "kind", "one", "ones", "area", "location", "entity",
}
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    kind: str
    formula: str
    information_layer: str


def _specs(
    family: str,
    layer: str,
    entries: Sequence[tuple[str, str, str]],
) -> list[FeatureSpec]:
    return [FeatureSpec(name, family, kind, formula, layer) for name, kind, formula in entries]


def feature_catalog(
    legacy_lexical_names: Sequence[str],
    dense_names: Sequence[str],
) -> list[FeatureSpec]:
    """Return the frozen scalar-feature catalog in output-column order."""

    specs: list[FeatureSpec] = []
    specs.extend(
        FeatureSpec(
            f"legacy_lexical__{name}",
            "historical_legacy_17d",
            "continuous",
            f"Exact stored legacy lexical column: {name}; token multiplicity retained.",
            "query_plus_corpus_static_exact",
        )
        for name in legacy_lexical_names
    )
    specs.extend(
        FeatureSpec(
            f"legacy_dense__{name}",
            "historical_dense_prototype_13d",
            "continuous",
            f"Exact stored frozen-corpus-prototype summary column: {name}.",
            "query_plus_frozen_corpus_prototypes",
        )
        for name in dense_names
    )
    specs.extend(
        FeatureSpec(
            f"unique_lexical__{name}",
            "bm25_aligned_unique_17d",
            "continuous",
            f"Legacy formula for {name}, recalculated on the unique analyzed-term set used by BM25.",
            "query_plus_corpus_static_exact",
        )
        for name in legacy_lexical_names
    )

    specs.extend(_specs("stage_c_collection", "query_plus_corpus_static_exact", [
        ("c_exact_idf_sum", "continuous", "Sum of full-index IDF over unique in-vocabulary BM25 terms."),
        ("c_exact_idf_median", "continuous", "Median full-index IDF over unique in-vocabulary BM25 terms."),
        ("c_exact_idf_iqr", "continuous", "IQR of full-index IDF over unique in-vocabulary BM25 terms."),
        ("c_exact_idf_cv", "continuous", "IDF standard deviation divided by mean over unique in-vocabulary terms."),
        ("c_exact_log_cf_mean", "continuous", "Mean log(1+collection frequency) over unique in-vocabulary terms."),
        ("c_exact_log_cf_std", "continuous", "SD of log(1+collection frequency) over unique in-vocabulary terms."),
        ("c_exact_ictf_mean", "continuous", "Mean log((total collection tokens+1)/(CF+1)) over unique in-vocabulary terms."),
        ("c_exact_ictf_max", "continuous", "Maximum log((total collection tokens+1)/(CF+1)) over unique in-vocabulary terms."),
        ("c_exact_scq_mean", "continuous", "Mean (1+log(CF))*IDF over unique in-vocabulary terms."),
        ("c_exact_scq_max", "continuous", "Maximum (1+log(CF))*IDF over unique in-vocabulary terms."),
        ("c_exact_cf_df_ratio_mean", "continuous", "Mean CF/DF over unique in-vocabulary terms."),
        ("c_exact_cf_df_ratio_std", "continuous", "SD of CF/DF over unique in-vocabulary terms."),
        ("c_exact_cf_df_ratio_max", "continuous", "Maximum CF/DF over unique in-vocabulary terms."),
        ("c_exact_df_ge_100k_ratio", "continuous", "Fraction of unique in-vocabulary terms with DF >= 100,000."),
        ("c_exact_df_ge_500k_ratio", "continuous", "Fraction of unique in-vocabulary terms with DF >= 500,000."),
        ("c_exact_rare_term_count_df_le_100", "continuous", "Count of unique in-vocabulary terms with DF <= 100."),
    ]))
    specs.extend(_specs("stage_c_scope", "query_conditioned_frozen_corpus_sample", [
        ("c_scope_log_any_docs", "continuous", "log(1+sample documents containing at least one effective term)."),
        ("c_scope_log_at_least_2_docs", "continuous", "log(1+sample documents containing at least two effective terms)."),
        ("c_scope_log_half_docs", "continuous", "log(1+sample documents containing at least ceil(m/2) effective terms)."),
        ("c_scope_log_all_docs", "continuous", "log(1+sample documents containing all m effective terms)."),
        ("c_scope_at_least_2_over_any", "continuous", "Sample scope(at least 2)/scope(any)."),
        ("c_scope_half_over_any", "continuous", "Sample scope(at least half)/scope(any)."),
        ("c_scope_all_over_any", "continuous", "Sample scope(all)/scope(any)."),
        ("c_scope_match_count_mean", "continuous", "Mean matched effective-term count conditional on any sample match."),
        ("c_scope_match_count_std", "continuous", "SD of matched effective-term count conditional on any sample match."),
        ("c_scope_match_count_entropy", "continuous", "Normalized entropy of matched-term counts among sample matches."),
        ("c_scope_zero_full_match", "binary", "1 when no sample document contains all effective terms; also the explicit sentinel for zero effective terms."),
    ]))
    specs.extend(_specs("stage_c_coherence", "query_conditioned_frozen_corpus_sample", [
        ("c_pair_jaccard_mean", "continuous", "Mean document-set Jaccard across sample-estimable effective-term pairs."),
        ("c_pair_jaccard_min", "continuous", "Minimum document-set Jaccard across effective-term pairs."),
        ("c_pair_jaccard_max", "continuous", "Maximum document-set Jaccard across effective-term pairs."),
        ("c_pair_jaccard_std", "continuous", "SD of document-set Jaccard across effective-term pairs."),
        ("c_pair_pmi_mean", "continuous", "Mean add-0.5 smoothed document-level PMI across sample-estimable effective-term pairs."),
        ("c_pair_pmi_min", "continuous", "Minimum add-0.5 smoothed document-level PMI across sample-estimable effective-term pairs."),
        ("c_pair_pmi_max", "continuous", "Maximum add-0.5 smoothed document-level PMI across sample-estimable effective-term pairs."),
        ("c_pair_npmi_mean", "continuous", "Mean add-0.5 smoothed normalized PMI across sample-estimable effective-term pairs."),
        ("c_pair_npmi_min", "continuous", "Minimum add-0.5 smoothed normalized PMI across sample-estimable effective-term pairs."),
        ("c_pair_overlap_mean", "continuous", "Mean overlap coefficient coDF/min(DF_i,DF_j) across sample-estimable term pairs."),
        ("c_pair_overlap_min", "continuous", "Minimum overlap coefficient across sample-estimable term pairs."),
        ("c_pair_zero_cooccurrence_ratio", "continuous", "Fraction of sample-estimable effective-term pairs with zero sample co-occurrence."),
        ("c_pair_estimable_ratio", "continuous", "Fraction of effective-term pairs for which both marginal sample DFs are nonzero."),
    ]))
    specs.extend(_specs("stage_c_term_impact", "query_plus_corpus_static_aggregate", [
        ("c_global_impact_mean", "continuous", "Mean sampled global BM25 contribution per effective term; heuristic IDF fallback when unsampled."),
        ("c_global_impact_std", "continuous", "SD across effective terms of sampled global BM25 mean contribution."),
        ("c_global_impact_min", "continuous", "Minimum sampled global BM25 mean contribution across effective terms."),
        ("c_global_impact_max", "continuous", "Maximum sampled global BM25 mean contribution across effective terms."),
        ("c_global_impact_p90", "continuous", "90th percentile sampled global BM25 mean contribution."),
        ("c_global_impact_cv", "continuous", "Global-impact SD divided by mean."),
        ("c_global_impact_max_share", "continuous", "Largest effective-term impact divided by total impact."),
        ("c_global_impact_entropy", "continuous", "Normalized entropy of effective-term global impacts."),
        ("c_global_impact_gt_5_ratio", "continuous", "Fraction of effective terms with global mean BM25 contribution > 5."),
        ("c_entity_impact_share", "continuous", "Fraction of total global impact assigned to capitalized/quoted entity terms."),
        ("c_relation_impact_share", "continuous", "Fraction of total global impact assigned to relation-hint terms."),
        ("c_numeric_impact_share", "continuous", "Fraction of total global impact assigned to numeric/year terms."),
        ("c_global_impact_sample_supported_ratio", "continuous", "Fraction of effective terms whose global mean contribution is estimated from at least one sampled document."),
    ]))
    specs.extend(_specs("stage_c_length", "query_conditioned_frozen_corpus_sample", [
        ("c_union_doc_length_mean", "continuous", "Mean BM25 document length among sample documents matching any effective term."),
        ("c_union_doc_length_std", "continuous", "SD of BM25 document length among sample documents matching any effective term."),
        ("c_union_doc_length_log_ratio", "continuous", "log(union-match mean length/full-corpus average length)."),
        ("c_union_doc_length_abs_log_ratio", "continuous", "Absolute union-match document-length log ratio."),
        ("c_full_match_doc_length_mean", "continuous", "Mean length of sample documents matching all effective terms; zero if none."),
        ("c_term_match_length_mean", "continuous", "Mean across terms of their sampled matching-document mean length."),
        ("c_term_match_length_std", "continuous", "SD across terms of their sampled matching-document mean length."),
    ]))
    specs.extend(_specs("stage_c_title_body", "query_conditioned_frozen_corpus_sample", [
        ("c_title_term_coverage", "continuous", "Fraction of effective terms observed in a sampled title."),
        ("c_body_term_coverage", "continuous", "Fraction of effective terms observed in a sampled body."),
        ("c_title_occurrence_share_mean", "continuous", "Mean titleDF/combinedDF across effective terms in sample."),
        ("c_title_occurrence_share_max", "continuous", "Maximum titleDF/combinedDF across effective terms in sample."),
        ("c_title_scope_log_any_docs", "continuous", "log(1+sample titles containing at least one effective term)."),
        ("c_title_scope_log_all_docs", "continuous", "log(1+sample titles containing all effective terms)."),
        ("c_title_scope_all_over_any", "continuous", "Title sample scope(all)/scope(any)."),
        ("c_title_pair_jaccard_mean", "continuous", "Mean title-document Jaccard across effective-term pairs."),
        ("c_title_pair_jaccard_min", "continuous", "Minimum title-document Jaccard across effective-term pairs."),
        ("c_entity_title_coverage", "continuous", "Fraction of full-index-known entity-anchor terms observed in a sampled title."),
    ]))
    specs.extend(_specs("stage_c_morph_competition", "query_plus_corpus_static_exact", [
        ("c_suffix_alt_known_ratio", "continuous", "Fraction of effective terms having a known deterministic suffix alternative; not a linguistic lemmatizer."),
        ("c_suffix_alt_df_advantage_mean", "continuous", "Mean max(0,log1p(max suffix-alt DF)-log1p(original DF))."),
        ("c_suffix_alt_df_advantage_max", "continuous", "Maximum corpus DF advantage of a deterministic suffix alternative."),
        ("c_suffix_alt_idf_gap_mean", "continuous", "Mean original IDF minus IDF of highest-DF known deterministic suffix alternative."),
        ("c_oov_suffix_recoverable_ratio", "continuous", "Fraction of OOV analyzed terms with a known deterministic suffix alternative."),
    ]))

    specs.extend(_specs("morphology_and_tokenizer", "query_only", [
        ("q_raw_word_count", "continuous", "Count of regex words before BM25 stopword removal."),
        ("q_analyzed_token_count", "continuous", "Count of BM25 analyzed tokens with multiplicity."),
        ("q_unique_analyzed_term_count", "continuous", "Count of unique BM25 analyzed terms."),
        ("q_duplicate_analyzed_ratio", "continuous", "1-unique analyzed terms/analyzed tokens."),
        ("q_stopword_removal_ratio", "continuous", "1-analyzed token count/raw word count."),
        ("q_character_count", "continuous", "Unicode character count after outer whitespace strip."),
        ("q_mean_raw_word_length", "continuous", "Mean raw regex-word length."),
        ("q_raw_word_length_std", "continuous", "SD of raw regex-word length."),
        ("q_hyphen_count", "continuous", "Count of ASCII and Unicode hyphen separators."),
        ("q_apostrophe_word_ratio", "continuous", "Fraction of raw words containing an apostrophe."),
        ("q_possessive_word_ratio", "continuous", "Fraction of raw words ending in apostrophe-s."),
        ("q_acronym_count", "continuous", "Count of all-capital raw tokens of length >=2."),
        ("q_alphanumeric_token_count", "continuous", "Count of raw tokens containing letters and digits."),
        ("q_inflection_suffix_ratio", "continuous", "Fraction of lowercase alphabetic words ending in s/es/ed/ing/ly."),
        ("q_analyzer_fragmentation_ratio", "continuous", "Analyzed-token count divided by raw-word count."),
    ]))
    specs.extend(_specs("relation_lexicalization", "query_plus_corpus_static_exact", [
        ("q_relation_hint_count", "continuous", "Count of analyzed tokens in the frozen relation lexicon or with verb-like suffix."),
        ("q_relation_hint_ratio", "continuous", "Relation-hint tokens divided by analyzed tokens."),
        ("q_relation_idf_mean", "continuous", "Mean full-index IDF of relation-hint terms."),
        ("q_relation_idf_max", "continuous", "Maximum full-index IDF of relation-hint terms."),
        ("q_relation_oov_ratio", "continuous", "OOV fraction among relation-hint terms."),
        ("q_passive_marker", "binary", "1 when a form of be is followed within two words by an ed/en-like word."),
        ("q_preposition_count", "continuous", "Count of frozen relational prepositions in raw lowercase words."),
        ("q_generic_relation_ratio", "continuous", "Fraction of relation-hint terms with DF >= 100,000."),
    ]))
    specs.extend(_specs("entity_and_anchor", "query_plus_corpus_static_exact", [
        ("q_capitalized_span_count", "continuous", "Count of contiguous capitalized-token spans, excluding an initial WH token."),
        ("q_capitalized_token_ratio", "continuous", "Capitalized entity tokens divided by raw words."),
        ("q_capitalized_span_max_words", "continuous", "Maximum number of words in a capitalized span."),
        ("q_quoted_span_count", "continuous", "Count of quoted spans."),
        ("q_numeric_token_count", "continuous", "Count of all-digit raw tokens."),
        ("q_year_token_count", "continuous", "Count of four-digit year tokens from 1500-2099."),
        ("q_parenthetical_span_count", "continuous", "Count of parenthetical spans."),
        ("q_entity_anchor_term_count", "continuous", "Count of unique capitalized or quoted analyzed terms."),
        ("q_entity_anchor_coverage", "continuous", "Corpus-vocabulary coverage of entity-anchor terms."),
        ("q_entity_anchor_idf_mean", "continuous", "Mean full-index IDF of entity-anchor terms."),
        ("q_entity_anchor_idf_min", "continuous", "Minimum full-index IDF of entity-anchor terms."),
        ("q_entity_anchor_idf_max", "continuous", "Maximum full-index IDF of entity-anchor terms."),
        ("q_entity_anchor_max_share", "continuous", "Largest entity-anchor IDF divided by entity-anchor IDF sum."),
    ]))
    specs.extend(_specs("lexical_economy", "query_plus_corpus_static_exact", [
        ("q_unique_term_ratio", "continuous", "Unique analyzed terms divided by analyzed tokens."),
        ("q_lexical_density", "continuous", "Analyzed tokens divided by raw words."),
        ("q_idf_per_raw_word", "continuous", "Unique-term IDF sum divided by raw-word count."),
        ("q_high_idf_term_ratio", "continuous", "Fraction of effective terms with IDF >= 5."),
        ("q_low_idf_term_ratio", "continuous", "Fraction of effective terms with IDF <= 2."),
        ("q_token_frequency_entropy", "continuous", "Normalized entropy of analyzed-token multiplicities."),
        ("q_max_token_frequency_share", "continuous", "Largest analyzed-token count divided by analyzed-token count."),
        ("q_punctuation_per_word", "continuous", "Non-alphanumeric nonspace character count divided by raw-word count."),
    ]))
    specs.extend(_specs("answer_type_and_constraints", "query_only", [
        ("q_wh_who", "binary", "1 when the first raw word is who/whom/whose."),
        ("q_wh_where", "binary", "1 when the first raw word is where."),
        ("q_wh_when", "binary", "1 when the first raw word is when."),
        ("q_wh_which", "binary", "1 when the first raw word is which."),
        ("q_wh_what", "binary", "1 when the first raw word is what."),
        ("q_wh_how", "binary", "1 when the first raw word is how."),
        ("q_how_many", "binary", "1 when the query contains the bigram how many/much."),
        ("q_what_year", "binary", "1 when a what/which year formulation occurs."),
        ("q_person_answer_cue", "binary", "1 when frozen person-answer cue words occur."),
        ("q_location_answer_cue", "binary", "1 when frozen location-answer cue words occur."),
        ("q_temporal_constraint", "binary", "1 when a frozen temporal cue or year occurs."),
        ("q_comparison_constraint", "binary", "1 when a frozen comparison cue occurs."),
        ("q_superlative_constraint", "binary", "1 when a frozen superlative cue occurs."),
        ("q_negation_constraint", "binary", "1 when a frozen negation cue occurs."),
        ("q_ordinal_constraint", "binary", "1 when an ordinal token occurs."),
        ("q_disjunction_constraint", "binary", "1 when or/either/neither occurs."),
        ("q_constraint_count", "continuous", "Count of temporal,numeric,comparison,superlative,negation,ordinal constraint types."),
    ]))
    specs.extend(_specs("ambiguity", "query_only", [
        ("q_pronoun_count", "continuous", "Count of frozen pronouns in raw lowercase words."),
        ("q_pronoun_ratio", "continuous", "Pronoun count divided by raw-word count."),
        ("q_demonstrative_count", "continuous", "Count of demonstratives/former/latter."),
        ("q_generic_head_count", "continuous", "Count of frozen generic head nouns."),
        ("q_entityless", "binary", "1 when no capitalized, quoted, numeric, or year anchor is detected."),
        ("q_parenthetical_disambiguation", "binary", "1 when at least one parenthetical span occurs."),
        ("q_alias_cue", "binary", "1 when known as/called/also named/aka occurs."),
        ("q_coordination_count", "continuous", "Count of and/or/both/either/neither coordinators."),
    ]))
    specs.extend(_specs("hotpot_multihop", "query_only", [
        ("q_bridge_marker_count", "continuous", "Count of frozen bridge-relation markers."),
        ("q_relative_clause_marker_count", "continuous", "Count of who/which/that/whose/where outside the first word."),
        ("q_clause_punctuation_count", "continuous", "Count of commas, semicolons, and colons."),
        ("q_multi_entity_marker", "binary", "1 when at least two capitalized/quoted spans are detected."),
        ("q_bridge_style_marker", "binary", "1 when a relative marker and at least one entity anchor co-occur."),
        ("q_comparison_style_marker", "binary", "1 when a comparison cue and at least two entity spans co-occur."),
        ("q_possessive_bridge_count", "continuous", "Count of possessive raw words, a common bridge-query construction."),
        ("q_relation_clause_density", "continuous", "Relation-hint plus relative-marker count divided by raw words."),
    ]))
    specs.extend(_specs("general_style_controls", "query_only", [
        ("q_question_mark", "binary", "1 when the stripped query ends in a question mark."),
        ("q_comma_count", "continuous", "Count of commas."),
        ("q_quote_character_count", "continuous", "Count of straight/curly double-quote characters."),
        ("q_parenthesis_character_count", "continuous", "Count of opening/closing parentheses."),
        ("q_lowercase_initial", "binary", "1 when the first cased character is lowercase."),
        ("q_keyword_style", "binary", "1 when no initial WH/auxiliary and no terminal question mark."),
        ("q_function_word_ratio", "continuous", "Frozen function-word count divided by raw-word count."),
        ("q_raw_type_token_ratio", "continuous", "Unique lowercase raw words divided by raw words."),
    ]))

    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        raise ValueError(f"Duplicate feature names: {duplicates}")
    return specs


def catalog_sha256(specs: Sequence[FeatureSpec]) -> str:
    payload = json.dumps([asdict(spec) for spec in specs], sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def deterministic_morph_variants(term: str) -> set[str]:
    """Small analyzer-level inflection set; no semantic or external alias resource."""

    variants: set[str] = set()
    if not term.isalpha() or len(term) < 3:
        return variants
    if term.endswith("ies") and len(term) > 4:
        variants.add(term[:-3] + "y")
    if term.endswith("es") and len(term) > 4:
        variants.add(term[:-2])
        variants.add(term[:-1])
    elif term.endswith("s") and len(term) > 3:
        variants.add(term[:-1])
    else:
        variants.add(term + "s")
        if term.endswith(("s", "x", "z", "ch", "sh")):
            variants.add(term + "es")
    if term.endswith("ied") and len(term) > 4:
        variants.add(term[:-3] + "y")
    if term.endswith("ed") and len(term) > 4:
        variants.add(term[:-2])
        variants.add(term[:-1])
    else:
        variants.add(term + "ed")
    if term.endswith("ing") and len(term) > 5:
        variants.add(term[:-3])
        variants.add(term[:-3] + "e")
    else:
        variants.add(term + "ing")
    variants.discard(term)
    return {value for value in variants if len(value) >= 3}


def query_term_universe(questions: Sequence[str]) -> list[str]:
    base = {term for question in questions for term in analyze_sqlite_bm25_text(question)}
    expanded = set(base)
    for term in base:
        expanded.update(deterministic_morph_variants(term))
    return sorted(expanded)


def split_title_body(source: str, text: str) -> tuple[str, str]:
    """Recover the verified HotpotQA title/body boundary from a chunk record."""

    if source.startswith("beir:"):
        return "", text
    if text == source:
        return source, ""
    if text.startswith(source + "\n"):
        return source, text[len(source) + 1 :]
    raise ValueError("Unrecognized title/body boundary")


def _read_exact_stats(
    connection: sqlite3.Connection,
    terms: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {term: position for position, term in enumerate(terms)}
    df = np.zeros(len(terms), dtype=np.int64)
    idf = np.zeros(len(terms), dtype=np.float64)
    cf = np.zeros(len(terms), dtype=np.int64)
    for part in _chunks(terms, 700):
        placeholders = ",".join("?" for _ in part)
        for term, value_df, value_idf in connection.execute(
            f"SELECT term, df, idf FROM term_stats WHERE term IN ({placeholders})", part
        ):
            position = index[str(term)]
            df[position] = int(value_df)
            idf[position] = float(value_idf)
    # The primary key is (term, vector_id), so each chunk is a bounded range scan.
    for part_number, part in enumerate(_chunks(terms, 120), start=1):
        placeholders = ",".join("?" for _ in part)
        for term, value_cf in connection.execute(
            f"SELECT term, SUM(tf) FROM postings WHERE term IN ({placeholders}) GROUP BY term", part
        ):
            cf[index[str(term)]] = int(value_cf)
        if part_number % 25 == 0:
            print(f"exact-cf chunks complete: {part_number}", flush=True)
    return df, idf, cf


def build_corpus_cache(
    *,
    questions: Sequence[str],
    index_path: Path,
    chunks_path: Path,
    offsets_path: Path,
    output_path: Path,
    sample_rows: int,
    sample_seed: int,
) -> dict[str, Any]:
    """Build exact marginal stats plus a fixed sampled corpus aggregate cache."""

    started = time.perf_counter()
    terms = query_term_universe(questions)
    uri = f"file:{index_path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    document_count = int(metadata["document_count"])
    average_document_length = float(metadata["average_document_length"])
    total_document_length = int(metadata["total_length"])
    exact_df, exact_idf, exact_cf = _read_exact_stats(connection, terms)
    connection.close()

    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    if offsets.shape != (document_count,):
        raise ValueError(f"Corpus offset rows differ: {offsets.shape} vs {document_count}")
    if not (0 < sample_rows <= document_count):
        raise ValueError("Invalid corpus sample size")
    rng = np.random.default_rng(sample_seed)
    sample_vector_ids = np.sort(rng.choice(document_count, size=sample_rows, replace=False))
    term_to_position = {term: index for index, term in enumerate(terms)}
    relevant = set(terms)
    postings: defaultdict[str, array[int]] = defaultdict(lambda: array("I"))
    title_postings: defaultdict[str, array[int]] = defaultdict(lambda: array("I"))
    sample_df = np.zeros(len(terms), dtype=np.int64)
    sample_cf = np.zeros(len(terms), dtype=np.int64)
    sample_title_df = np.zeros(len(terms), dtype=np.int64)
    sample_body_df = np.zeros(len(terms), dtype=np.int64)
    sample_length_sum = np.zeros(len(terms), dtype=np.float64)
    sample_impact_sum = np.zeros(len(terms), dtype=np.float64)
    sample_impact_sq_sum = np.zeros(len(terms), dtype=np.float64)
    sample_impact_max = np.zeros(len(terms), dtype=np.float64)
    sample_doc_lengths = np.zeros(sample_rows, dtype=np.uint16)
    k1 = 1.5
    b = 0.75

    with chunks_path.open("rb") as handle:
        for sample_id, vector_id_value in enumerate(sample_vector_ids):
            vector_id = int(vector_id_value)
            handle.seek(int(offsets[vector_id]))
            row = json.loads(handle.readline().decode("utf-8"))
            if int(row["vector_id"]) != vector_id:
                raise ValueError(f"Corpus offset mismatch at vector_id={vector_id}")
            text = str(row.get("text", ""))
            source = str(row.get("source", ""))
            try:
                title_text, body = split_title_body(source, text)
            except ValueError as error:
                raise ValueError(f"Unrecognized title/body boundary at vector_id={vector_id}") from error
            combined_tokens = list(analyze_sqlite_bm25_text(text))
            title_terms = set(analyze_sqlite_bm25_text(title_text)) & relevant
            body_terms = set(analyze_sqlite_bm25_text(body)) & relevant
            counts = Counter(term for term in combined_tokens if term in relevant)
            document_length = len(combined_tokens)
            if document_length > np.iinfo(np.uint16).max:
                raise ValueError("Sample document length exceeds uint16")
            sample_doc_lengths[sample_id] = document_length
            norm = k1 * (1.0 - b + b * document_length / average_document_length)
            for term, tf in counts.items():
                position = term_to_position[term]
                postings[term].append(sample_id)
                sample_df[position] += 1
                sample_cf[position] += int(tf)
                sample_length_sum[position] += document_length
                impact = exact_idf[position] * (tf * (k1 + 1.0)) / (tf + norm)
                sample_impact_sum[position] += impact
                sample_impact_sq_sum[position] += impact * impact
                sample_impact_max[position] = max(sample_impact_max[position], impact)
            for term in title_terms:
                position = term_to_position[term]
                title_postings[term].append(sample_id)
                sample_title_df[position] += 1
            for term in body_terms:
                sample_body_df[term_to_position[term]] += 1
            if (sample_id + 1) % 25000 == 0:
                elapsed = time.perf_counter() - started
                print(f"corpus sample: {sample_id + 1}/{sample_rows} rows ({elapsed:.1f}s)", flush=True)

    def flatten(source: Mapping[str, array[int]]) -> tuple[np.ndarray, np.ndarray]:
        lengths = np.asarray([len(source.get(term, ())) for term in terms], dtype=np.int64)
        value_offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(lengths)])
        flat = np.empty(int(value_offsets[-1]), dtype=np.uint32)
        cursor = 0
        for term in terms:
            values = source.get(term)
            if values:
                block = np.frombuffer(values, dtype=np.uint32)
                flat[cursor : cursor + len(block)] = block
                cursor += len(block)
        return value_offsets, flat

    posting_offsets, posting_doc_ids = flatten(postings)
    title_offsets, title_doc_ids = flatten(title_postings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        terms=np.asarray(terms, dtype=np.str_),
        exact_df=exact_df,
        exact_idf=exact_idf,
        exact_cf=exact_cf,
        sample_df=sample_df,
        sample_cf=sample_cf,
        sample_title_df=sample_title_df,
        sample_body_df=sample_body_df,
        sample_length_sum=sample_length_sum,
        sample_impact_sum=sample_impact_sum,
        sample_impact_sq_sum=sample_impact_sq_sum,
        sample_impact_max=sample_impact_max,
        posting_offsets=posting_offsets,
        posting_doc_ids=posting_doc_ids,
        title_offsets=title_offsets,
        title_doc_ids=title_doc_ids,
        sample_doc_lengths=sample_doc_lengths,
        sample_vector_ids=np.asarray(sample_vector_ids, dtype=np.int64),
        document_count=np.asarray([document_count], dtype=np.int64),
        total_document_length=np.asarray([total_document_length], dtype=np.int64),
        average_document_length=np.asarray([average_document_length], dtype=np.float64),
        sample_rows=np.asarray([sample_rows], dtype=np.int64),
        sample_seed=np.asarray([sample_seed], dtype=np.int64),
    )
    del offsets
    elapsed = time.perf_counter() - started
    return {
        "terms": len(terms),
        "known_terms": int(np.sum(exact_df > 0)),
        "exact_postings_aggregated": int(np.sum(exact_df)),
        "sample_rows": sample_rows,
        "sample_postings": int(len(posting_doc_ids)),
        "sample_title_postings": int(len(title_doc_ids)),
        "elapsed_seconds": elapsed,
    }


class CorpusCache:
    def __init__(self, path: Path):
        stored = np.load(path, allow_pickle=False)
        self._stored = stored
        self.terms = np.asarray(stored["terms"], dtype=np.str_)
        self.term_to_position = {str(term): i for i, term in enumerate(self.terms)}
        for name in (
            "exact_df", "exact_idf", "exact_cf", "sample_df", "sample_cf",
            "sample_title_df", "sample_body_df", "sample_length_sum",
            "sample_impact_sum", "sample_impact_sq_sum", "sample_impact_max",
            "posting_offsets", "posting_doc_ids", "title_offsets", "title_doc_ids",
            "sample_doc_lengths",
        ):
            setattr(self, name, np.asarray(stored[name]))
        self.document_count = int(stored["document_count"][0])
        self.total_document_length = int(stored["total_document_length"][0])
        self.average_document_length = float(stored["average_document_length"][0])
        self.sample_rows = int(stored["sample_rows"][0])

    def close(self) -> None:
        self._stored.close()

    def position(self, term: str) -> int | None:
        return self.term_to_position.get(term)

    def postings(self, term: str, *, title: bool = False) -> np.ndarray:
        position = self.position(term)
        if position is None:
            return np.empty(0, dtype=np.uint32)
        offsets = self.title_offsets if title else self.posting_offsets
        values = self.title_doc_ids if title else self.posting_doc_ids
        return values[int(offsets[position]) : int(offsets[position + 1])]

    def values(self, terms: Sequence[str], field: str) -> np.ndarray:
        source = np.asarray(getattr(self, field))
        return np.asarray(
            [source[position] if (position := self.position(term)) is not None else 0 for term in terms],
            dtype=np.float64,
        )


def _safe_mean(values: Sequence[float] | np.ndarray) -> float:
    array_value = np.asarray(values, dtype=np.float64)
    return float(np.mean(array_value)) if len(array_value) else 0.0


def _safe_std(values: Sequence[float] | np.ndarray) -> float:
    array_value = np.asarray(values, dtype=np.float64)
    return float(np.std(array_value)) if len(array_value) else 0.0


def _safe_min(values: Sequence[float] | np.ndarray) -> float:
    array_value = np.asarray(values, dtype=np.float64)
    return float(np.min(array_value)) if len(array_value) else 0.0


def _safe_max(values: Sequence[float] | np.ndarray) -> float:
    array_value = np.asarray(values, dtype=np.float64)
    return float(np.max(array_value)) if len(array_value) else 0.0


def _safe_percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    array_value = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array_value, q)) if len(array_value) else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _normalized_entropy(weights: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    probabilities = values / values.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(probabilities)))


def _capitalized_spans(words: Sequence[str]) -> list[list[str]]:
    spans: list[list[str]] = []
    current: list[str] = []
    for index, word in enumerate(words):
        is_initial_wh = index == 0 and word.casefold() in WH_WORDS
        is_capitalized = bool(CAPITALIZED_RE.fullmatch(word) or ACRONYM_RE.fullmatch(word))
        if is_capitalized and not is_initial_wh:
            current.append(word)
        elif current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return spans


def _aligned_lexical_values(
    question: str,
    cache: CorpusCache,
) -> list[float]:
    terms = sorted(set(analyze_sqlite_bm25_text(question)))
    known = [term for term in terms if cache.position(term) is not None and cache.values([term], "exact_df")[0] > 0]
    idf = cache.values(known, "exact_idf")
    df = cache.values(known, "exact_df")
    denominator = max(1, len(terms))
    raw_words = RAW_WORD_RE.findall(question)
    numeric = [word.casefold() for word in raw_words if word.isdigit()]
    years = [word.casefold() for word in raw_words if YEAR_RE.fullmatch(word)]
    capitalized = [
        word.casefold() for word in raw_words[1:]
        if word[:1].isupper() and not word.isupper()
    ]
    quoted = [
        token for phrase in QUOTED_RE.findall(question)
        for token in analyze_sqlite_bm25_text(phrase)
    ]

    def max_idf(candidates: Sequence[str]) -> float:
        return _safe_max(cache.values(candidates, "exact_idf"))

    return [
        _ratio(len(known), denominator),
        _ratio(len(terms) - len(known), denominator),
        _safe_mean(idf),
        _safe_std(idf),
        _safe_min(idf),
        _safe_max(idf),
        _safe_percentile(idf, 90),
        _ratio(_safe_max(idf), float(np.sum(idf))),
        _safe_mean(np.log1p(df)),
        _safe_std(np.log1p(df)),
        float(np.mean(df <= 10)) if len(df) else 0.0,
        float(np.mean(df <= 100)) if len(df) else 0.0,
        float(np.mean(df <= 1000)) if len(df) else 0.0,
        max_idf(numeric),
        max_idf(years),
        max_idf(capitalized),
        max_idf(quoted),
    ]


def _scope_values(arrays: Sequence[np.ndarray], term_count: int) -> tuple[np.ndarray, np.ndarray]:
    nonempty = [values for values in arrays if len(values)]
    if not nonempty or term_count <= 0:
        return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.int64)
    docs, counts = np.unique(np.concatenate(nonempty), return_counts=True)
    return docs, counts


def _pair_values(arrays: Sequence[np.ndarray], sample_rows: int) -> dict[str, float]:
    jaccard: list[float] = []
    pmi: list[float] = []
    npmi: list[float] = []
    overlap: list[float] = []
    zero = 0
    total_pairs = 0
    estimable_pairs = 0
    for left in range(len(arrays)):
        for right in range(left + 1, len(arrays)):
            total_pairs += 1
            a = arrays[left]
            b = arrays[right]
            if len(a) == 0 or len(b) == 0:
                continue
            estimable_pairs += 1
            co = int(np.intersect1d(a, b, assume_unique=True).size) if len(a) and len(b) else 0
            if co == 0:
                zero += 1
            union = len(a) + len(b) - co
            jaccard.append(_ratio(co, union))
            # Haldane-Anscombe smoothing on all four document-contingency cells.
            n11 = co + 0.5
            n10 = len(a) - co + 0.5
            n01 = len(b) - co + 0.5
            n00 = sample_rows - len(a) - len(b) + co + 0.5
            total = n11 + n10 + n01 + n00
            p11 = n11 / total
            pa = (n11 + n10) / total
            pb = (n11 + n01) / total
            value_pmi = math.log(p11 / (pa * pb))
            probability = min(1.0 - 1e-12, p11)
            value_npmi = value_pmi / max(1e-12, -math.log(probability))
            pmi.append(value_pmi)
            npmi.append(value_npmi)
            overlap.append(_ratio(co, min(len(a), len(b))))
    pairs = len(jaccard)
    return {
        "mean_jaccard": _safe_mean(jaccard),
        "min_jaccard": _safe_min(jaccard),
        "max_jaccard": _safe_max(jaccard),
        "std_jaccard": _safe_std(jaccard),
        "mean_pmi": _safe_mean(pmi),
        "min_pmi": _safe_min(pmi),
        "max_pmi": _safe_max(pmi),
        "mean_npmi": _safe_mean(npmi),
        "min_npmi": _safe_min(npmi),
        "mean_overlap": _safe_mean(overlap),
        "min_overlap": _safe_min(overlap),
        "zero_ratio": _ratio(zero, pairs),
        "estimable_ratio": _ratio(estimable_pairs, total_pairs),
    }


def _question_features(question: str, cache: CorpusCache) -> dict[str, float]:
    raw_words = RAW_WORD_RE.findall(question)
    lower_words = [word.casefold() for word in raw_words]
    tokens = list(analyze_sqlite_bm25_text(question))
    token_counts = Counter(tokens)
    unique_terms = sorted(token_counts)
    known_terms = [
        term for term in unique_terms
        if (position := cache.position(term)) is not None and cache.exact_df[position] > 0
    ]
    oov_terms = [term for term in unique_terms if term not in set(known_terms)]
    idf = cache.values(known_terms, "exact_idf")
    df = cache.values(known_terms, "exact_df")
    cf = cache.values(known_terms, "exact_cf")
    spans = _capitalized_spans(raw_words)
    quoted_spans = QUOTED_RE.findall(question)
    entity_terms = {
        term
        for span in spans
        for word in span
        for term in analyze_sqlite_bm25_text(word)
    } | {
        term
        for phrase in quoted_spans
        for term in analyze_sqlite_bm25_text(phrase)
    }
    numeric_terms = {word.casefold() for word in raw_words if word.isdigit()}
    relation_tokens = [
        term for term in tokens
        if term in RELATION_HINTS or term.endswith(("ed", "ing", "ion", "er"))
    ]
    relation_terms = sorted(set(relation_tokens))
    relation_idf = cache.values(
        [term for term in relation_terms if cache.position(term) is not None], "exact_idf"
    )
    relation_df = cache.values(
        [term for term in relation_terms if cache.position(term) is not None], "exact_df"
    )

    result: dict[str, float] = {}
    # Stage C collection statistics.
    ictf = np.log((cache.total_document_length + 1.0) / (cf + 1.0)) if len(cf) else np.empty(0)
    scq = (1.0 + np.log(np.maximum(cf, 1.0))) * idf if len(cf) else np.empty(0)
    cf_df = cf / np.maximum(df, 1.0) if len(cf) else np.empty(0)
    result.update({
        "c_exact_idf_sum": float(np.sum(idf)),
        "c_exact_idf_median": _safe_percentile(idf, 50),
        "c_exact_idf_iqr": _safe_percentile(idf, 75) - _safe_percentile(idf, 25),
        "c_exact_idf_cv": _ratio(_safe_std(idf), _safe_mean(idf)),
        "c_exact_log_cf_mean": _safe_mean(np.log1p(cf)),
        "c_exact_log_cf_std": _safe_std(np.log1p(cf)),
        "c_exact_ictf_mean": _safe_mean(ictf),
        "c_exact_ictf_max": _safe_max(ictf),
        "c_exact_scq_mean": _safe_mean(scq),
        "c_exact_scq_max": _safe_max(scq),
        "c_exact_cf_df_ratio_mean": _safe_mean(cf_df),
        "c_exact_cf_df_ratio_std": _safe_std(cf_df),
        "c_exact_cf_df_ratio_max": _safe_max(cf_df),
        "c_exact_df_ge_100k_ratio": float(np.mean(df >= 100000)) if len(df) else 0.0,
        "c_exact_df_ge_500k_ratio": float(np.mean(df >= 500000)) if len(df) else 0.0,
        "c_exact_rare_term_count_df_le_100": float(np.sum(df <= 100)) if len(df) else 0.0,
    })

    arrays = [cache.postings(term) for term in known_terms]
    union_docs, match_counts = _scope_values(arrays, len(known_terms))
    half = max(1, math.ceil(len(known_terms) / 2))
    any_count = len(union_docs)
    at_least_2 = int(np.sum(match_counts >= 2)) if len(match_counts) else 0
    half_count = int(np.sum(match_counts >= half)) if len(match_counts) else 0
    all_count = int(np.sum(match_counts >= len(known_terms))) if len(match_counts) and known_terms else 0
    count_hist = np.bincount(match_counts, minlength=len(known_terms) + 1)[1:] if len(match_counts) else np.empty(0)
    result.update({
        "c_scope_log_any_docs": math.log1p(any_count),
        "c_scope_log_at_least_2_docs": math.log1p(at_least_2),
        "c_scope_log_half_docs": math.log1p(half_count),
        "c_scope_log_all_docs": math.log1p(all_count),
        "c_scope_at_least_2_over_any": _ratio(at_least_2, any_count),
        "c_scope_half_over_any": _ratio(half_count, any_count),
        "c_scope_all_over_any": _ratio(all_count, any_count),
        "c_scope_match_count_mean": _safe_mean(match_counts),
        "c_scope_match_count_std": _safe_std(match_counts),
        "c_scope_match_count_entropy": _normalized_entropy(count_hist),
        "c_scope_zero_full_match": float(all_count == 0),
    })
    pairs = _pair_values(arrays, cache.sample_rows)
    result.update({
        "c_pair_jaccard_mean": pairs["mean_jaccard"],
        "c_pair_jaccard_min": pairs["min_jaccard"],
        "c_pair_jaccard_max": pairs["max_jaccard"],
        "c_pair_jaccard_std": pairs["std_jaccard"],
        "c_pair_pmi_mean": pairs["mean_pmi"],
        "c_pair_pmi_min": pairs["min_pmi"],
        "c_pair_pmi_max": pairs["max_pmi"],
        "c_pair_npmi_mean": pairs["mean_npmi"],
        "c_pair_npmi_min": pairs["min_npmi"],
        "c_pair_overlap_mean": pairs["mean_overlap"],
        "c_pair_overlap_min": pairs["min_overlap"],
        "c_pair_zero_cooccurrence_ratio": pairs["zero_ratio"],
        "c_pair_estimable_ratio": pairs["estimable_ratio"],
    })

    sample_df = cache.values(known_terms, "sample_df")
    sample_impact_sum = cache.values(known_terms, "sample_impact_sum")
    impacts = np.where(sample_df > 0, sample_impact_sum / np.maximum(sample_df, 1.0), idf)
    total_impact = float(np.sum(impacts))
    impact_by_term = {term: float(value) for term, value in zip(known_terms, impacts)}
    result.update({
        "c_global_impact_mean": _safe_mean(impacts),
        "c_global_impact_std": _safe_std(impacts),
        "c_global_impact_min": _safe_min(impacts),
        "c_global_impact_max": _safe_max(impacts),
        "c_global_impact_p90": _safe_percentile(impacts, 90),
        "c_global_impact_cv": _ratio(_safe_std(impacts), _safe_mean(impacts)),
        "c_global_impact_max_share": _ratio(_safe_max(impacts), total_impact),
        "c_global_impact_entropy": _normalized_entropy(impacts),
        "c_global_impact_gt_5_ratio": float(np.mean(impacts > 5.0)) if len(impacts) else 0.0,
        "c_entity_impact_share": _ratio(sum(impact_by_term.get(t, 0.0) for t in entity_terms), total_impact),
        "c_relation_impact_share": _ratio(sum(impact_by_term.get(t, 0.0) for t in relation_terms), total_impact),
        "c_numeric_impact_share": _ratio(sum(impact_by_term.get(t, 0.0) for t in numeric_terms), total_impact),
        "c_global_impact_sample_supported_ratio": float(np.mean(sample_df > 0)) if len(sample_df) else 0.0,
    })

    union_lengths = cache.sample_doc_lengths[union_docs] if len(union_docs) else np.empty(0)
    full_docs = union_docs[match_counts >= len(known_terms)] if len(match_counts) and known_terms else np.empty(0, dtype=np.uint32)
    full_lengths = cache.sample_doc_lengths[full_docs] if len(full_docs) else np.empty(0)
    length_sum = cache.values(known_terms, "sample_length_sum")
    term_lengths = np.where(sample_df > 0, length_sum / np.maximum(sample_df, 1.0), cache.average_document_length)
    union_mean = _safe_mean(union_lengths)
    log_ratio = math.log(max(union_mean, 1e-12) / cache.average_document_length) if union_mean > 0 else 0.0
    result.update({
        "c_union_doc_length_mean": union_mean,
        "c_union_doc_length_std": _safe_std(union_lengths),
        "c_union_doc_length_log_ratio": log_ratio,
        "c_union_doc_length_abs_log_ratio": abs(log_ratio),
        "c_full_match_doc_length_mean": _safe_mean(full_lengths),
        "c_term_match_length_mean": _safe_mean(term_lengths),
        "c_term_match_length_std": _safe_std(term_lengths),
    })

    title_df = cache.values(known_terms, "sample_title_df")
    body_df = cache.values(known_terms, "sample_body_df")
    title_arrays = [cache.postings(term, title=True) for term in known_terms]
    title_docs, title_counts = _scope_values(title_arrays, len(known_terms))
    title_any = len(title_docs)
    title_all = int(np.sum(title_counts >= len(known_terms))) if len(title_counts) and known_terms else 0
    title_pairs = _pair_values(title_arrays, cache.sample_rows)
    entity_known = [term for term in sorted(entity_terms) if term in known_terms]
    entity_title_df = cache.values(entity_known, "sample_title_df")
    result.update({
        "c_title_term_coverage": float(np.mean(title_df > 0)) if len(title_df) else 0.0,
        "c_body_term_coverage": float(np.mean(body_df > 0)) if len(body_df) else 0.0,
        "c_title_occurrence_share_mean": _safe_mean(title_df / np.maximum(sample_df, 1.0)),
        "c_title_occurrence_share_max": _safe_max(title_df / np.maximum(sample_df, 1.0)),
        "c_title_scope_log_any_docs": math.log1p(title_any),
        "c_title_scope_log_all_docs": math.log1p(title_all),
        "c_title_scope_all_over_any": _ratio(title_all, title_any),
        "c_title_pair_jaccard_mean": title_pairs["mean_jaccard"],
        "c_title_pair_jaccard_min": title_pairs["min_jaccard"],
        "c_entity_title_coverage": float(np.mean(entity_title_df > 0)) if len(entity_title_df) else 0.0,
    })

    alt_known = 0
    alt_advantages: list[float] = []
    alt_idf_gaps: list[float] = []
    for term in known_terms:
        original_position = cache.position(term)
        assert original_position is not None
        alternatives = [
            value for value in deterministic_morph_variants(term)
            if (position := cache.position(value)) is not None and cache.exact_df[position] > 0
        ]
        if not alternatives:
            continue
        alt_known += 1
        alt_df_values = cache.values(alternatives, "exact_df")
        best_index = int(np.argmax(alt_df_values))
        best = alternatives[best_index]
        best_position = cache.position(best)
        assert best_position is not None
        advantage = max(0.0, math.log1p(cache.exact_df[best_position]) - math.log1p(cache.exact_df[original_position]))
        alt_advantages.append(advantage)
        alt_idf_gaps.append(float(cache.exact_idf[original_position] - cache.exact_idf[best_position]))
    recoverable = 0
    for term in oov_terms:
        if any(
            (position := cache.position(value)) is not None and cache.exact_df[position] > 0
            for value in deterministic_morph_variants(term)
        ):
            recoverable += 1
    result.update({
        "c_suffix_alt_known_ratio": _ratio(alt_known, len(known_terms)),
        "c_suffix_alt_df_advantage_mean": _safe_mean(alt_advantages),
        "c_suffix_alt_df_advantage_max": _safe_max(alt_advantages),
        "c_suffix_alt_idf_gap_mean": _safe_mean(alt_idf_gaps),
        "c_oov_suffix_recoverable_ratio": _ratio(recoverable, len(oov_terms)),
    })

    # Query-only morphology and lexical economy.
    word_lengths = [len(word) for word in raw_words]
    apostrophes = sum("'" in word or "’" in word for word in raw_words)
    possessives = sum(word.casefold().endswith(("'s", "’s")) for word in raw_words)
    suffix_words = sum(
        word.isalpha() and word.casefold().endswith(("s", "es", "ed", "ing", "ly"))
        for word in raw_words
    )
    punctuation_count = sum(not character.isalnum() and not character.isspace() for character in question)
    result.update({
        "q_raw_word_count": float(len(raw_words)),
        "q_analyzed_token_count": float(len(tokens)),
        "q_unique_analyzed_term_count": float(len(unique_terms)),
        "q_duplicate_analyzed_ratio": 1.0 - _ratio(len(unique_terms), len(tokens)),
        "q_stopword_removal_ratio": 1.0 - _ratio(len(tokens), len(raw_words)),
        "q_character_count": float(len(question.strip())),
        "q_mean_raw_word_length": _safe_mean(word_lengths),
        "q_raw_word_length_std": _safe_std(word_lengths),
        "q_hyphen_count": float(sum(question.count(mark) for mark in ("-", "–", "—"))),
        "q_apostrophe_word_ratio": _ratio(apostrophes, len(raw_words)),
        "q_possessive_word_ratio": _ratio(possessives, len(raw_words)),
        "q_acronym_count": float(sum(bool(ACRONYM_RE.fullmatch(word)) for word in raw_words)),
        "q_alphanumeric_token_count": float(sum(bool(ALPHANUMERIC_RE.search(word)) for word in raw_words)),
        "q_inflection_suffix_ratio": _ratio(suffix_words, len(raw_words)),
        "q_analyzer_fragmentation_ratio": _ratio(len(tokens), len(raw_words)),
        "q_relation_hint_count": float(len(relation_tokens)),
        "q_relation_hint_ratio": _ratio(len(relation_tokens), len(tokens)),
        "q_relation_idf_mean": _safe_mean(relation_idf),
        "q_relation_idf_max": _safe_max(relation_idf),
        "q_relation_oov_ratio": _ratio(sum(cache.position(t) is None or cache.exact_df[cache.position(t)] == 0 for t in relation_terms), len(relation_terms)),
        "q_passive_marker": float(any(
            lower_words[i] in {"is", "was", "were", "been", "be", "being"}
            and any(word.endswith(("ed", "en")) for word in lower_words[i + 1 : i + 3])
            for i in range(len(lower_words))
        )),
        "q_preposition_count": float(sum(word in {"of", "in", "at", "by", "for", "from", "with", "to"} for word in lower_words)),
        "q_generic_relation_ratio": float(np.mean(relation_df >= 100000)) if len(relation_df) else 0.0,
    })

    entity_known_idf = cache.values([term for term in sorted(entity_terms) if term in known_terms], "exact_idf")
    result.update({
        "q_capitalized_span_count": float(len(spans)),
        "q_capitalized_token_ratio": _ratio(sum(len(span) for span in spans), len(raw_words)),
        "q_capitalized_span_max_words": float(max((len(span) for span in spans), default=0)),
        "q_quoted_span_count": float(len(quoted_spans)),
        "q_numeric_token_count": float(sum(word.isdigit() for word in raw_words)),
        "q_year_token_count": float(sum(bool(YEAR_RE.fullmatch(word)) for word in raw_words)),
        "q_parenthetical_span_count": float(min(question.count("("), question.count(")"))),
        "q_entity_anchor_term_count": float(len(entity_terms)),
        "q_entity_anchor_coverage": _ratio(len(entity_known_idf), len(entity_terms)),
        "q_entity_anchor_idf_mean": _safe_mean(entity_known_idf),
        "q_entity_anchor_idf_min": _safe_min(entity_known_idf),
        "q_entity_anchor_idf_max": _safe_max(entity_known_idf),
        "q_entity_anchor_max_share": _ratio(_safe_max(entity_known_idf), float(np.sum(entity_known_idf))),
        "q_unique_term_ratio": _ratio(len(unique_terms), len(tokens)),
        "q_lexical_density": _ratio(len(tokens), len(raw_words)),
        "q_idf_per_raw_word": _ratio(float(np.sum(idf)), len(raw_words)),
        "q_high_idf_term_ratio": float(np.mean(idf >= 5.0)) if len(idf) else 0.0,
        "q_low_idf_term_ratio": float(np.mean(idf <= 2.0)) if len(idf) else 0.0,
        "q_token_frequency_entropy": _normalized_entropy(list(token_counts.values())),
        "q_max_token_frequency_share": _ratio(max(token_counts.values(), default=0), len(tokens)),
        "q_punctuation_per_word": _ratio(punctuation_count, len(raw_words)),
    })

    first = lower_words[0] if lower_words else ""
    joined = " ".join(lower_words)
    temporal = bool(set(lower_words) & TEMPORAL_HINTS or any(YEAR_RE.fullmatch(word) for word in raw_words))
    numeric = any(word.isdigit() for word in raw_words)
    comparison = bool(set(lower_words) & COMPARISON_HINTS)
    superlative = bool(set(lower_words) & SUPERLATIVE_HINTS)
    negation = bool(set(lower_words) & NEGATIONS)
    ordinal = any(ORDINAL_RE.fullmatch(word) for word in raw_words)
    result.update({
        "q_wh_who": float(first in {"who", "whom", "whose"}),
        "q_wh_where": float(first == "where"),
        "q_wh_when": float(first == "when"),
        "q_wh_which": float(first == "which"),
        "q_wh_what": float(first == "what"),
        "q_wh_how": float(first == "how"),
        "q_how_many": float("how many" in joined or "how much" in joined),
        "q_what_year": float("what year" in joined or "which year" in joined),
        "q_person_answer_cue": float(bool(set(lower_words) & PERSON_HINTS)),
        "q_location_answer_cue": float(bool(set(lower_words) & LOCATION_HINTS)),
        "q_temporal_constraint": float(temporal),
        "q_comparison_constraint": float(comparison),
        "q_superlative_constraint": float(superlative),
        "q_negation_constraint": float(negation),
        "q_ordinal_constraint": float(ordinal),
        "q_disjunction_constraint": float(bool(set(lower_words) & {"or", "either", "neither"})),
        "q_constraint_count": float(sum((temporal, numeric, comparison, superlative, negation, ordinal))),
    })

    pronoun_count = sum(word in PRONOUNS for word in lower_words)
    bridge_count = sum(word in BRIDGE_HINTS for word in lower_words)
    relative_count = sum(word in {"who", "which", "that", "whose", "where"} for word in lower_words[1:])
    result.update({
        "q_pronoun_count": float(pronoun_count),
        "q_pronoun_ratio": _ratio(pronoun_count, len(raw_words)),
        "q_demonstrative_count": float(sum(word in DEMONSTRATIVES for word in lower_words)),
        "q_generic_head_count": float(sum(word in GENERIC_HEADS for word in lower_words)),
        "q_entityless": float(not spans and not quoted_spans and not numeric),
        "q_parenthetical_disambiguation": float("(" in question and ")" in question),
        "q_alias_cue": float(any(phrase in joined for phrase in ("known as", "also named", "also called", "aka"))),
        "q_coordination_count": float(sum(word in {"and", "or", "both", "either", "neither"} for word in lower_words)),
        "q_bridge_marker_count": float(bridge_count),
        "q_relative_clause_marker_count": float(relative_count),
        "q_clause_punctuation_count": float(sum(question.count(mark) for mark in (",", ";", ":"))),
        "q_multi_entity_marker": float(len(spans) + len(quoted_spans) >= 2),
        "q_bridge_style_marker": float(relative_count > 0 and bool(entity_terms)),
        "q_comparison_style_marker": float(comparison and len(spans) + len(quoted_spans) >= 2),
        "q_possessive_bridge_count": float(possessives),
        "q_relation_clause_density": _ratio(len(relation_tokens) + relative_count, len(raw_words)),
    })

    first_cased = next((character for character in question if character.isalpha()), "")
    result.update({
        "q_question_mark": float(question.strip().endswith("?")),
        "q_comma_count": float(question.count(",")),
        "q_quote_character_count": float(sum(question.count(mark) for mark in ('"', "“", "”"))),
        "q_parenthesis_character_count": float(question.count("(") + question.count(")")),
        "q_lowercase_initial": float(bool(first_cased and first_cased.islower())),
        "q_keyword_style": float(first not in WH_WORDS | QUESTION_AUXILIARIES and not question.strip().endswith("?")),
        "q_function_word_ratio": _ratio(sum(word in FUNCTION_WORDS for word in lower_words), len(raw_words)),
        "q_raw_type_token_ratio": _ratio(len(set(lower_words)), len(lower_words)),
    })
    return result


def extract_feature_matrix(
    *,
    questions: Sequence[str],
    legacy_lexical: np.ndarray,
    legacy_dense: np.ndarray,
    legacy_lexical_names: Sequence[str],
    dense_names: Sequence[str],
    cache: CorpusCache,
) -> tuple[np.ndarray, list[FeatureSpec]]:
    specs = feature_catalog(legacy_lexical_names, dense_names)
    expected_new_names = [
        spec.name for spec in specs
        if not spec.name.startswith(("legacy_lexical__", "legacy_dense__", "unique_lexical__"))
    ]
    rows: list[list[float]] = []
    for index, question in enumerate(questions):
        computed = _question_features(question, cache)
        if set(computed) != set(expected_new_names):
            missing = sorted(set(expected_new_names) - set(computed))
            extra = sorted(set(computed) - set(expected_new_names))
            raise ValueError(f"Feature implementation mismatch: missing={missing}, extra={extra}")
        values = (
            np.asarray(legacy_lexical[index], dtype=np.float64).tolist()
            + np.asarray(legacy_dense[index], dtype=np.float64).tolist()
            + _aligned_lexical_values(question, cache)
            + [float(computed[name]) for name in expected_new_names]
        )
        if len(values) != len(specs):
            raise ValueError("Feature row dimension differs from catalog")
        rows.append(values)
        if (index + 1) % 500 == 0:
            print(f"features: {index + 1}/{len(questions)} queries", flush=True)
    matrix = np.asarray(rows, dtype=np.float64)
    return matrix, specs
