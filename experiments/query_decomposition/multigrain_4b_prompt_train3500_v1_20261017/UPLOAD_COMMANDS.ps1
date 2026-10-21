# Run manually in PowerShell. Uses the existing SSH host entry; no remote connection has been tested here.
$promptPackagePath = 'C:\Users\12442\Desktop\RAG_pipeine\RAG_pipeline\outputs\query_decomposition\multigrain_4b_prompt_train3500_v1_20261017\original4b_prompt100_remote_package.zip'
if (-not (Test-Path -LiteralPath $promptPackagePath -PathType Leaf)) {
    throw "Package not found: $promptPackagePath"
}
scp $promptPackagePath 'cluster1.comp.anu.edu.au:~/original4b_prompt100_remote_package.zip'
if ($LASTEXITCODE -ne 0) {
    throw 'Upload failed. Keep the local package and report the SSH error.'
}
