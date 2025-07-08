<#
.SYNOPSIS
  Prepends each file's content with a "// filename" comment (excluding this script) 
  and copies the combined output to the clipboard.

.DESCRIPTION
  Place this script in the folder you want to process, cd there, then run:
    .\PrintFolder.ps1

  It will:
    • Enumerate all files in the current directory except itself
    • For each file, emit a line "// filename.ext", then its contents, then a blank line
    • Print everything to the console
    • Copy the full concatenation to the Windows clipboard
#>

# Figure out our own filename so we can skip it
$selfName = Split-Path -Leaf $PSCommandPath

# Get all files in the current directory, excluding this script
$files = Get-ChildItem -File |
         Where-Object { $_.Name -ne $selfName } |
         Sort-Object Name

# Build the combined output
$output = foreach ($file in $files) {
    "// $($file.Name)"
    Get-Content -Raw -Path $file.FullName
    ""  # blank line between files
}

# Print to console
$output

# Copy to clipboard
$output | Set-Clipboard

Write-Host "`n✅ Processed $($files.Count) file(s) and copied to clipboard." -ForegroundColor Green
