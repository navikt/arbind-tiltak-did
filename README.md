# Tiltaksnedgang - forskjeller-i-forskjeller-analyse

Dette prosjektet estimerer hvordan reduksjonen i norske arbeidsmarkedstiltak i
2025 påvirket Navs arbeidsindikatorer. Det støtter både vanlige
forskjeller-i-forskjeller-analyser (DiD) og trippel-diff-analyser på region- og
enhetsnivå.

Katalogen i `src/diff-in-diff/configs/catalog.yml` definerer alle 
analyser. Legg til/fjern varianter for å kun kjøre de spesifikke analysene. 

## Kom i gang

### Forutsetninger

- Python 3.14 eller nyere
- [uv](https://docs.astral.sh/uv/)
- [just](https://just.systems/)
- Quarto, bare ved rendering av rapporten
- Tilgang til Navs BigQuery-datakilde, ved oppdatering av produksjonsdata

Installer utviklingsmiljøet uten å endre låste avhengighetsversjoner:

```sh
uv sync --dev
```

Kjør testene:

```sh
just test
```

`data/`, `outputs/` og `quarto/` er med hensikt ignorert av Git. En fersk klone
kan kjøre testene, men kan ikke kjøre en produksjonsanalyse før nødvendige
datafiler er lagt til eller oppdatert.

## Kjør en analyse

Katalogen i `src/diff-in-diff/configs/catalog.yml` definerer alle tillatte
analyser. Genererte konfigurasjons-ID-er identifiserer både en kjøring og dens
utdataplassering. Se
[`src/diff-in-diff/configs/README.md`](src/diff-in-diff/configs/README.md) for
alle tilgjengelige valg og hvordan du legger til en ny matrise.

Kjør gjeldende standardkonfigurasjon:

```sh
just did
```

Kjør én bestemt konfigurasjon:

```sh
just did config='did--midl-lonnstilskudd--alle--regioner--kontinuerlig--vektet'
```

Kjør bare basis- og foretrukne regresjoner for hver katalogkonfigurasjon:

```sh
just run-regressions
```

Denne kommandoen kjører ikke tester, bootstrap-inferens, hendelsesstudier,
leave-one-out-diagnostikk, rapportgenerering eller Quarto-rendering.

Generer rapporter fra lagrede analyseresultater, og render deretter
Quarto-boken:

```sh
just generate-report
just render
```

## Legg til eller endre analysematriser

`catalog.yml` er sannhetskilden for gyldige analyser. Hver oppføring i `matrix`
utvider de angitte dimensjonene til genererte kjørings-ID-er:

- `design`: `did` eller `triple_diff`
- `measures`, `groups` og `levels`
- `treatments`: `diskret` eller `kontinuerlig`
- `outcomes`: `indikator`, `faktisk` eller `forventet`
- `weighting`: `unweighted` eller `personer`

Slik prøver du en ny matrise:

1. Legg til eller endre en matriseoppføring i
   `src/diff-in-diff/configs/catalog.yml`.
2. Sørg for at input-stiene kan genereres fra filmalen i `defaults`, og at
   tilhørende databatch finnes lokalt.
3. Kjør `just test` for å validere katalogutvidelsen og analyseprogrammet.
4. Kjør ønsket genererte ID med `just did config='<generert-id>'`, eller alle
   konfigurasjoner med `just run-regressions`.
5. Gjennomgå de genererte resultatene før matrisen tas i bruk som en støttet
   analyse.

Katalogtestene validerer med hensikt konfigurasjonsstrukturen i stedet for å
kreve ignorerte produksjonsdatafiler. Dermed kan matriseendringer vurderes i en
fersk klone; datakompatibilitet hører til dataoppdateringsarbeidsflyten.

## Oppdater en databatch

Produksjonsinput er ikke versjonert i Git. Oppdater alle kilder før
analyseperioden utvides eller en ny produksjonsbatch kjøres.

### Indikatordata fra BigQuery

Uttrekksskriptene spør tabellen i prosjektet `arbeidsindikator-prod-51bc`, angitt
av tabell-URI-en i `src/fetch_data/bq_client.py`. De bruker
[Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc).
Autentiser i tråd med teamets GCP-policy, for eksempel med:

```sh
gcloud auth application-default login
```

Den autentiserte identiteten må kunne kjøre spørringen og lese den konfigurerte
tabellen. Tilgangstildeling og valg av legitimasjon håndteres utenfor dette
repositoriet; ikke legg legitimasjon eller token inn i arbeidstreet.

Oppdater regionale, enhetsvise og nasjonale indikatoruttrekk, og bygg de
kombinerte veiledningsgruppene på nytt:

```sh
just fetch-all
```

For én gruppe eller ett nivå, bruk `just fetch-fylke '<gruppe>'` eller
`just fetch-enhet '<gruppe>'`. Uttrekksskriptene skriver ignorerte filer under
`data/input/indikatorer/` og `data/input/personer/`. Enhetsanalyser krever også
oppdatert `data/input/støttedata/enhetsmapping.json`.

### Tiltaksdata

Legg oppdaterte kildeuttrekk i:

```text
data/støttedata/tiltak-region.json
data/støttedata/tiltak-region-type.json
```

Opprett deretter analyseinput:

```sh
just tiltak-to-csv
```

Dette skriver `data/input/tiltak/alle-tiltak.csv` og
`data/input/tiltak/midl.-lønnstilskudd.csv`.

### Utvid analyseperioden

Når oppdaterte filer inneholder de nye månedene:

1. Oppdater `defaults.period` eller en bestemt matriseoppførings `period`-overstyring
   i `catalog.yml`.
2. Kjør `just test`.
3. Kjør den relevante analysen eller `just run-regressions`.
4. Generer og render rapporter når resultatene er klare for gjennomgang.

Når data for mars 2026 er tilgjengelig, oppdateres for eksempel `data_end` til
`202603` der perioden skal gjelde. Hendelsesstudievinduet er fast fra -24 til +6
måneder; senere observasjoner utvider det ikke automatisk.

## Kart over repositoriet

| Sti | Formål |
|---|---|
| `src/diff-in-diff/` | Analyseprogram, katalog, regresjonsestimatorer og rapportgenerering |
| `src/fetch_data/` | BigQuery-uttrekk og sammenslåing av veiledningsgrupper |
| `src/utils/` | Operasjonelle verktøy for datakonvertering |
| `tests/` | Deterministiske enhets- og pipelinetester |
| `analysedesign.md` | Statistisk design og modellforutsetninger |
| `justfile` | Vanlige utviklings-, uttrekks-, analyse- og rapportkommandoer |
