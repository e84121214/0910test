$ErrorActionPreference = 'Stop'
$log = Join-Path $PSScriptRoot 'render_word.log'
$docx = Join-Path $env:TEMP 'admissions_draft.docx'
$pdf = Join-Path $env:TEMP 'admissions_draft.pdf'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'admissions_draft.docx') -Destination $docx -Force
"copied" | Set-Content -LiteralPath $log -Encoding ascii
$word = New-Object -ComObject Word.Application
"created word" | Add-Content -LiteralPath $log -Encoding ascii
$word.Visible = $false
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3
try {
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    "opened" | Add-Content -LiteralPath $log -Encoding ascii
    $doc.SaveAs2($pdf, 17)
    "exported" | Add-Content -LiteralPath $log -Encoding ascii
    $doc.Close($false)
    "closed doc" | Add-Content -LiteralPath $log -Encoding ascii
} finally {
    $word.Quit()
    "quit" | Add-Content -LiteralPath $log -Encoding ascii
}
$outDir = Join-Path $PSScriptRoot 'render_admissions_word'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$dest = Join-Path $outDir 'admissions_draft.pdf'
Copy-Item -LiteralPath $pdf -Destination $dest -Force
Write-Output $dest
