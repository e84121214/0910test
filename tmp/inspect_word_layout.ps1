$ErrorActionPreference = 'Stop'
$docx = Join-Path $env:TEMP 'admissions_draft.docx'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    $doc.Repaginate()
    $pages = $doc.ComputeStatistics(2)
    $words = $doc.ComputeStatistics(0)
    $chars = $doc.ComputeStatistics(3)
    $tables = $doc.Tables.Count
    $paras = $doc.Paragraphs.Count
    [PSCustomObject]@{
        Pages = $pages
        Words = $words
        Characters = $chars
        Paragraphs = $paras
        Tables = $tables
    } | ConvertTo-Json
    $doc.Close($false)
} finally {
    $word.Quit()
}
