$ErrorActionPreference = 'Stop'
$eval300Package = 'C:\Users\12442\Desktop\RAG_pipeine\RAG_pipeline\outputs\query_decomposition\hotpot_formal_eval_v1\complete_pipeline_comparison_v1_20261020\remote_reference_qdmr300\reference_qdmr300_remote_package.zip'
& scp $eval300Package 'u7830488@cluster1.comp.anu.edu.au:/home/remote/u7830488/densex_workspace/'
if ($LASTEXITCODE -ne 0) { throw 'scp failed' }
