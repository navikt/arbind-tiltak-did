# https://just.systems

# Hvis ingen kommando vis alle tilgjengelige oppskrifter
default:
    @just --list

# Klargjør prosjektet ved å installere `prek` og oppdatere avhengigheter fra malen
prepare:
    uv run --only-dev prek install
    uv lock --upgrade

# Fiks feil og formater kode med ruff
fix:
    uv run --only-dev ruff check --fix .
    uv run --only-dev ruff format .

# Sjekk at alt koden ser bra ut og er klar for å legges til i git
lint:
    uv run --only-dev prek run --all-files --color always

# Lag et preview med Quarto
preview:
    uv run --group quarto quarto preview .

# Bygg Quarto-boken
render:
    cd quarto && uv run --group quarto quarto render

# Publiser Quarto-boken til Datamarkedsplassen (krever TEAM_TOKEN_PROD, kjør `just render` først)
publish:
    uv run src/utils/publish.py

# Kjør DiD-analyse for alle katalogkonfigurasjoner
run-all new_only='':
    cd src/diff-in-diff && uv run python run_analysis.py --all {{new_only}}

# Generer Quarto-rapporter og data-kapitler fra lagrede analyseresultater
generate-report:
    cd src/diff-in-diff && uv run python generate_reports.py --all

# Kjør standard DiD-analyse for en katalog-ID
did config='did--midl-lonnstilskudd--alle--regioner--kontinuerlig':
    cd src/diff-in-diff && uv run python run_analysis.py --config {{config}}

# Kjør trippel-diff-analyse for regioner
triple-diff-regioner config='triple-diff--midl-lonnstilskudd--regioner--kontinuerlig':
    cd src/diff-in-diff && uv run python run_analysis.py --config {{config}}

# Kjør trippel-diff-analyse for enheter
triple-diff-enheter config='triple-diff--midl-lonnstilskudd--enheter--kontinuerlig':
    cd src/diff-in-diff && uv run python run_analysis.py --config {{config}}

# Hent indikatordata på enhetsnivå fra BigQuery (f.eks. `just fetch-enhet 'Alle'`)
fetch-enhet group:
    uv run python src/fetch_data/get_enhet_data.py "{{group}}"

# Hent indikatordata på fylkesnivå fra BigQuery (f.eks. `just fetch-fylke 'Alle'`)
fetch-fylke group:
    uv run python src/fetch_data/get_fylke_data.py "{{group}}"

# Slå sammen veiledning-grupper på enhetsnivå
merge-veiledning-enhet:
    uv run python src/fetch_data/merge_veiledning_enhet.py

# Konverter tiltak-JSON til CSV-filer i data/input/tiltak/
tiltak-to-csv:
    uv run python src/utils/tiltak_json_to_csv.py

# Eksporter behandlingsvariabel per region til Excel
export-treatment:
    uv run python src/utils/export_treatment_excel.py

# Slå sammen veiledning-grupper på fylkesnivå
merge-veiledning-fylke:
    uv run python src/fetch_data/merge_veiledning.py

# Hent all indikatordata fra BigQuery og slå sammen veiledning-grupper
fetch-all:
    uv run python src/fetch_data/get_fylke_data.py 'Alle' 'Standard' 'Spesielt tilpasset' 'Situasjonsbestemt'
    uv run python src/fetch_data/get_enhet_data.py 'Alle' 'Standard' 'Spesielt tilpasset' 'Situasjonsbestemt'
    uv run python src/fetch_data/get_landet_data.py 'Alle' 'Standard' 'Spesielt tilpasset' 'Situasjonsbestemt'
    just merge-veiledning-fylke
    just merge-veiledning-enhet

# Sjekk etter sårbarheter i Python-avhengigheter
audit:
    uv run --all-groups --with pip-audit pip-audit --local

# Generer data-seksjonen i Quarto-boken
data-report:
    cd src/diff-in-diff && uv run python generate_data_report.py

# Kjør tester
test:
    uv run --dev pytest tests/ -v

# Oppdater Python og pre-commit avhengigheter
update:
    uv lock --upgrade
    uv run prek auto-update
