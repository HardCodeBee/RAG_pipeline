"""Offline checks for the new task's actual format/model/resume boundaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
TEST_TEMP=HERE/"local_validation"/"tmp"
TEST_TEMP.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("base100_contracts",HERE/"remote/code/contracts.py")
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Contracts(unittest.TestCase):
    def test_new_array_is_valid_and_old_goal_object_is_rejected(self):
        self.assertTrue(c.parse_array('["Who wrote Book R?"]')["format_valid"])
        self.assertFalse(c.parse_array('{"steps":[{"id":1,"request":"Who wrote Book R?"}],"goal":1}')["format_valid"])

    def test_empty_array_is_valid_but_fenced_or_mixed_output_is_not_repaired(self):
        self.assertTrue(c.parse_array('[]')["format_valid"])
        self.assertFalse(c.parse_array('```json\n[]\n```')["format_valid"])
        self.assertFalse(c.parse_array('["Valid?", 3]')["format_valid"])

    def test_reference_flag_preserves_raw_question(self):
        parsed=c.parse_array('["When was #1 born?"]')
        self.assertTrue(parsed["format_valid"])
        self.assertTrue(parsed["explicit_references"])
        self.assertEqual(parsed["parsed_questions"],["When was #1 born?"])

    def test_four_outputs_for_same_query_are_retained(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            row={"id":"same-id","query":"Original?"}
            for arm in c.ARMS:
                c.save_json(Path(directory)/"cases"/arm/"001.json",{**row,"arm":arm})
            self.assertEqual(len(c.read_cases(directory,[row])),4)

    def test_resume_rejects_other_query_under_existing_index(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            c.save_json(Path(directory)/"cases/v8/001.json",{"id":"q1","query":"Wrong?","arm":"v8"})
            with self.assertRaises(AssertionError): c.read_cases(directory,[{"id":"q1","query":"Correct?"}])

    def test_failed_generation_is_counted_in_full_denominator(self):
        r={"arm":"v8","generation_error":"OOM","format_valid":False,"truncated":False}
        s=c.summary([r])
        self.assertEqual(s["arms"]["v8"]["planned"],100)
        self.assertEqual(s["arms"]["v8"]["completed_attempts"],1)
        self.assertEqual(s["arms"]["v8"]["generation_errors"],1)

    def test_model_state_rejects_lora_or_other_revision(self):
        config=json.loads((HERE/"remote/base_runtime.json").read_text(encoding="utf-8"))
        c.validate_config(config)
        for field,value in (("adapter_path","checkpoint-000148"),("revision","wrong"),("expected_adapter_parameters",66060288)):
            bad={**config,field:value}
            with self.assertRaises(AssertionError): c.validate_config(bad)

    def test_all_query_fields_except_id_and_query_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            path=Path(directory)/"queries.jsonl"
            records=[{"id":str(i),"query":"Question?","answer":"leak"} for i in range(100)]
            path.write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf-8")
            with self.assertRaises(AssertionError): c.queries(path)


if __name__ == "__main__": unittest.main()
