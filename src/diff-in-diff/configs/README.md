# Konfigurering av analysematriser

`catalog.yml` er sannhetskilden for hvilke forskjeller-i-forskjeller- (DiD) og
trippel-diff-analyser som kjøres. `config_matrix.py` leser katalogen ved
oppstart og utvider hver oppføring i `matrix` til én eller flere konkrete
konfigurasjoner.

Legg til eller endre analyser her, ikke ved å hardkode nye kombinasjoner i
analyseprogrammet.

## Katalogens deler

### `defaults`

`defaults` inneholder verdier som gjelder når de ikke overstyres i en
matriseoppføring.

| Felt | Betydning |
|---|---|
| `period` | Standard analyseperiode: `treatment_start`, `data_start` og `data_end`, alle på `YYYYMM`-format. |
| `denominator_definitions` | Tillatte nevnere for kontinuerlig behandling. Hver har en stabil `id` og en visnings-`label`. |
| `control_regions` | Standard kontrollregioner for diskrete DiD-analyser. |
| `enhet_mapping_file` | Sti til JSON-filen som kobler Nav-enheter til regioner. Kreves for `enhet`-nivå. |
| `indicators` | Indikatorene som inkluderes i hver analyse. Nøkkelen er filnavnet uten `.csv`, og verdien er en menneskelesbar beskrivelse. |
| `indicator_file_template` | Mal for indikatorfilene. Variablene `{level_directory}`, `{group_directory}`, `{outcome_prefix}` og `{indicator}` fylles ut automatisk. |

### Tillatte dimensjonsverdier

Disse seksjonene definerer hvilke verdier en matrise kan referere til:

| Seksjon | Tillatte verdier | Betydning |
|---|---|---|
| `measures` | `midl-lonnstilskudd`, `alle-tiltak` | Nøkkel og filnavn for tiltaksserien. |
| `groups` | `alle`, `personer_i_tiltak`, `situasjonsbestemt`, `spesielt-tilpasset`, `standard`, `veiledning-kombinert` | Innsatsgruppe eller samlet gruppe for indikator- og persondata. |
| `levels` | `region`, `enhet` | Analysenivå. `enhet` bruker indikatorer på langt format og `enhet_mapping_file`. |
| `treatments` | `diskret`, `kontinuerlig` | Diskret behandling bruker kontrollregioner; kontinuerlig behandling bruker nedbyggingsintensitet og nevnerdefinisjonene. |
| `outcomes` | `indikator`, `faktisk`, `forventet` | Hvilken indikatorserie som analyseres. De to siste får henholdsvis prefikset `faktisk_` og `forventet_` i filstien. |

For å introdusere en ny verdi må den først legges til i den riktige seksjonen,
og nødvendige inputfiler må være tilgjengelige lokalt.

## `matrix`: kombinasjoner som skal kjøre

Hver oppføring i `matrix` velger verdier fra dimensjonene over. For vanlige
DiD-analyser lages det én konfigurasjon for hvert produkt av `measures`,
`groups`, `levels`, `treatments`, `outcomes` og `weighting`.

Eksempel:

```yaml
- design: did
  measures: [midl-lonnstilskudd]
  groups: [alle, standard]
  levels: [region]
  treatments: [kontinuerlig]
  outcomes: [indikator, faktisk]
  weighting: [personer]
```

Eksemplet gir fire konfigurasjoner: to grupper ganger to utfall.

### Felles felt

| Felt | Påkrevd | Verdier / virkning |
|---|---|---|
| `design` | Ja | `did` for vanlig DiD eller `triple_diff` for trippel-diff. |
| `measures` | Ja | En eller flere nøkler fra `measures`. |
| `levels` | Ja | En eller flere verdier fra `levels`. |
| `treatments` | Ja | En eller flere verdier fra `treatments`. |
| `outcomes` | Ja | En eller flere verdier fra `outcomes`. |
| `weighting` | Nei | `unweighted` eller `personer`. Uten feltet brukes `[unweighted]`. |
| `period` | Nei | Overstyrer deler av `defaults.period` for denne oppføringen. |

`weighting: personer` krever personantallsfilen som svarer til den valgte
gruppen og nivået. Vektede analyser bruker personantall som regresjonsvekt.

### Vanlig DiD: `design: did`

En DiD-oppføring krever i tillegg:

| Felt | Påkrevd | Betydning |
|---|---|---|
| `groups` | Ja | En eller flere grupper fra `groups`. |
| `control_regions` | Nei | Overstyrer `defaults.control_regions` for diskrete analyser. Har ingen virkning for kontinuerlig behandling. |

For diskret behandling må den effektive listen med kontrollregioner være
ikke-tom. Kontinuerlig behandling bruker i stedet
`defaults.denominator_definitions`.

### Trippel-diff: `design: triple_diff`

En trippel-diff-oppføring erstatter `groups` med:

| Felt | Påkrevd | Betydning |
|---|---|---|
| `treated_group` | Ja | Gruppen som skal behandles i trippel-diff-interaksjonen. |
| `control_group` | Ja | Sammenligningsgruppen i trippel-diff-interaksjonen. |

Begge gruppene må være nøkler fra `groups`. Når `weighting: personer` brukes,
opprettes det personantallsreferanser for både behandlet og kontrollgruppe.

## Genererte ID-er

En vanlig DiD-konfigurasjon får ID-en:

```text
did--<measure>--<group>--<regioner|enheter>--<treatment|outcome-treatment>[--vektet]
```

En trippel-diff-konfigurasjon får ID-en:

```text
triple-diff--<measure>--<regioner|enheter>--<treatment|outcome-treatment>[--vektet]
```

Suffikset `--vektet` finnes bare når `weighting: personer` er valgt. Bruk
ID-en med:

```sh
just did config='<generert-id>'
```

Eller kjør bare regresjonene for alle definerte konfigurasjoner:

```sh
just run-regressions
```

## Anbefalt arbeidsflyt for en ny matrise

1. Kontroller at nødvendige datafiler er oppdatert. Se rot-`README.md` for
   oppdatering av indikator- og tiltaksdata.
2. Legg til eller juster én oppføring i `matrix`.
3. Bruk eksplisitt `period` bare når oppføringen skal avvike fra
   `defaults.period`.
4. Kjør `just test`. Dette verifiserer at katalogen kan utvides og at
   standardkonfigurasjonen fortsatt er gyldig.
5. Kjør den nye genererte ID-en med `just did config='<generert-id>'`.
6. Gjennomgå resultatene før kombinasjonen tas i bruk i en større kjøring.

Unngå å endre betydningen av en eksisterende ID uten å dokumentere det i
endringen. Katalogen bestemmer både analyseinnhold og utdataorganisering.

Denne dokumentasjonen er KI-generert. 