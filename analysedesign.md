# Analysedesign: Effekten av tiltaksnedbygging på Navs arbeidsindikator

## Bakgrunn og problemstilling

Nav (Arbeids- og velferdsetaten) er organisert i 12 regioner som opererer selvstendig. Fra og med juni 2025 ble bruken av **arbeidsmarkedstiltak** — kurs, lønnstilskudd, praksisplasser og lignende virkemidler — gradvis redusert som følge av tidligere overforbruk. Nedgangen var ujevnt fordelt: noen regioner ble i liten grad berørt, andre opplevde svært store kutt.

Nav måler effekten av sitt arbeid gjennom månedlige **arbeidsindikatorer** per region. Disse er standardiserte scorer som sammenligner regionale sysselsettingsutfall med hva som forventes basert på lokal brukersammensetning og arbeidsmarked. Indikatorene er beregnet per registreringskohort med ulike oppfølgingshorisonter:

| Indikator | Hva måles |
|:----------|:----------|
| atid3 | Gjennomsnittlig ukentlig arbeidstid neste 3 måneder |
| jobb3 | I jobb (mer enn 19 timer i uken) etter 3 måneder |

Analysen søker å besvare spørsmålene:
-	Hva er effekten av redusert tiltaksbruk på arbeidsindikatoren? 
-	I hvilken grad førte nedgangen i bruken av midlertidig lønnstilskudd fra juni 2025 til endringer i deltakergrunnlaget i tiltakene, målt ved arbeidsindikatorens forventningskomponent?
-	Er effekten forskjellig for ulike grupper som de med nedsatt arbeidsevne, arbeidssøkere og tiltaksdeltakere?

---

## Behandlingsvariabelen

En binær behandlingsvariabel (behandlet/ikke behandlet) ville ikke fanget den gradvise og regionsspesifikke karakteren til nedbyggingen. I stedet bruker vi en **kontinuerlig, tidsvarierende nedbyggingsintensitet**:

$$D_{it} = \max\!\left(0,\; \frac{\text{ref}_i - \text{tiltak}_{it}}{\text{ref}_i}\right) \quad \text{for } t \geq \text{juni 2025}, \quad D_{it} = 0 \text{ ellers}$$

Her er $\text{ref}_i$ referansenivået for antall tiltaksplasser i region $i$ rett før behandlingsperioden. I analysen brukes **siste pre-periode-måneden** ($t = -1$, mai 2025) som referanse. Variabelen går fra 0 (ingen reduksjon) til 1 (total eliminering) og er definert som null i alle måneder før juni 2025. Den observerte variasjonen i nedbyggingsintensitet strekker seg fra om lag 14 % for de minst berørte regionene til om lag 70 % for de hardest rammede, og intensiteten øker over tid gjennom post-perioden.

### Valget av referansepunkt

Alternativt kunne man brukt **toppen i pre-perioden** («peak») som referanse. Dette ble vurdert og forkastet. I noen regioner stammer det historiske toppunktet fra et tidligere syklisk oppsving i tiltaksbruken — et oppsving som fulgte ordinær arbeidsmarkedsutvikling og som ikke har noe med det administrative nedbygg-sjokket fra 2025 å gjøre. Å normalisere mot en slik topp ville overvurdert nedbyggingsintensiteten i disse regionene og skapt kunstige variasjoner på tvers av regioner. Referansepunktet $t = -1$ fanger derimot den faktiske størrelsen på kuttet fra det nivået som faktisk ble redusert. Robusthetssjekker med peak-normalisering gir i praksis lignende resultater.

Dette gir to fordeler: det tillater en **dose-respons-tolkning** (større kutt gir større effekt), og det er mer informativt enn en binær indikator for analyser der behandlingsintensiteten varierer.

---

## DiD-regresjonsmodellen

Analysen bruker et **difference-in-differences-design** (DiD) med paneldata. Grunnmodellen er:

$$Y_{it} = \beta \cdot D_{it} + \alpha_i + \gamma_t + \varepsilon_{it}$$

Der:
- $Y_{it}$ er indikatorscoren for region $i$ i måned $t$
- $D_{it}$ er nedbyggingsintensiteten
- $\alpha_i$ er regionale faste effekter (absorberer tidsinvariante forskjeller mellom regioner)
- $\gamma_t$ er år-måned faste effekter (absorberer landsdekkende trender, inkludert den generelle nedgangen i indikatorene)
- $\beta$ er den kausale effektestimaten

To modellspesifikasjoner estimeres:

| Modell | Data | Faste effekter | Formål |
|:-------|:-----|:---------------|:-------|
| Basis | Ujustert indikator | Region + år-måned | Grunnmodell |
| Foretrukket | Sesongjustert indikator | Region + år-måned | Håndterer sesongmønstre i data |

Den **foretrukne modellen** bruker en sesongjustert versjon av indikatoren. Sesongjusteringen utføres i dataforberedelsen: for hver region og hver kalendermåned beregnes gjennomsnittlig avvik fra regionens pre-periode-gjennomsnitt, og dette trekkes fra observasjonene. Fordi sesongjusteringen er basert utelukkende på pre-periode-data, introduseres ingen fremtidig informasjon. Begge modeller bruker identisk regresjonsoppsett (region + år-måned faste effekter); de skiller seg kun i inputdata. Tester viser at resultatene er nær identiske, noe som bekrefter at regionsspesifikke sesongmønstre ikke er en vesentlig trussel mot identifikasjonen.

---

## Identifikasjonsantagelsen

Den sentrale antagelsen i DiD er **parallelle trender**: fraværende tiltaksnedbyggingen ville regionene ha fulgt parallelle baner i indikatorene. Fordi år-måned-dummyene absorberer landsdekkende trender, reduseres dette i praksis til en antagelse om at *avvik fra den felles trenden* ikke ville variert systematisk med nedbyggingsintensiteten.

### Eksogeitet: budsjettoverskridelse som kvasi-tilfeldig sjokk

En sentral forutsetning er at nedbyggingsintensiteten er eksogen — det vil si ukorrelert med de potensielle utfallene. Denne antagelsen er godt støttet av mekanismen bak sjokket: overforbruket av tiltaksmidler ble oppdaget sent på grunn av forsinkelser i administrativ og finansiell rapportering. Alle regioner strammet inn, men regionene med størst budsjettoverskridelse måtte kutte mest. Intensiteten av kuttet er dermed i hovedsak en funksjon av en pre-determinert finansiell overskridelsesstørrelse — ikke av løpende arbeidsmarkedsforhold. Dette er nær en naturlig variasjon som kan betraktes som kvasi-tilfeldig betinget på region- og tidspunkt-faste effekter.

### Fraværet av spillover-effekter (SUTVA)

SUTVA-antagelsen (Stable Unit Treatment Value Assumption) krever at behandlingen i én region ikke påvirker utfallene i andre regioner. I dette tilfellet er antagelsen godt begrunnet: nedbyggingen innebærer ikke at det opprettes nye tiltaksplasser andre steder — det er ingen nye midler å fordele. Arbeidere som ikke lenger får tilbud om tiltak, vil typisk forbli registrert i sin hjemregion uten å flytte. Etterspørselen etter tiltak er heller ikke fluid nok til at reduksjoner i én region skaper kompenserende tilstrømning til andre regioner.

### Utfallskohorten

Arbeidsindikatoren er ikke beregnet utelukkende for tiltaksdeltakere, men for en bredere kohort av alle registrerte Nav-brukere (eller spesifiserte undergrupper) som potensielt kvalifiserer for tiltak. Dette betyr at endringer i hvem som faktisk mottar tiltak ikke direkte endrer hvem som inngår i nevneren. Sammensetningsskjevhet som følge av at tiltaksnedbygging endrer «tiltakspopulasjonen» er dermed ikke en vesentlig trussel.

Et analytisk gunstig forvarsel er at regionene med størst kutt hadde en tendens til å *overutøve* i perioden før nedbyggingen. Dette gjør det lite sannsynlig at de hardest rammede regionene allerede var i selvstendig nedgang, noe som ellers ville vært en trussel mot identifikasjonen. Det innebærer dessuten at effektestimatene trolig er **konservative**: en region som presterte over forventning ville uansett hatt noe tilbaketrekning mot gjennomsnittet.

---

## Hendelsesstudie

For å underbygge identifikasjonsantagelsen og kartlegge effektenes dynamikk estimeres en **hendelsesstudie** (event study):

$$Y_{it} = \sum_{\tau \neq -1} \beta_\tau \cdot s_i \cdot \mathbf{1}\{\text{rel\_periode}_{it} = \tau\} + \alpha_i + \gamma_t + \varepsilon_{it}$$

Der $s_i$ er den maksimale nedbyggingsintensiteten for region $i$ over post-behandlingsperioden (et tidsinvariant mål på «behandlingsdosen»), og $\tau$ er antall måneder relativt til behandlingsstart. Referanseperioden er måneden umiddelbart før behandlingsstart ($\tau = -1$, mai 2025), slik at alle estimater er relative til dette tidspunktet. Vinduet strekker seg 24 måneder bakover og 6 måneder fremover.

### Interaksjon med tidsinvariant intensitet

En viktig designbeslutning er at $s_i$ er *tidsinvariant* — den er basert på den maksimale post-behandlingsintensiteten, ikke den løpende $D_{it}$. Dette er bevisst. I en hendelsesstudie ønsker vi å la de tidsvarierende effektene komme frem gjennom $\beta_\tau$-koeffisientene alene; å bruke $D_{it}$ i stedet for $s_i$ ville innebære at behandlingsintensiteten selv varierer over $\tau$, noe som blander dose med timing og gjør koeffisientene vanskeligere å tolke. Ved å holde $s_i$ fast gir $\beta_\tau$ en ren tidsprofil: *«for en region med full nedbygging, hva er den estimerte effekten $\tau$ måneder fra behandlingsstart?»*

### Normalisering og referanseperiode

$\tau = -1$ (mai 2025) utelates og fungerer som referansenivå. Alle $\beta_\tau$ tolkes som avvik fra dette tidspunktet. Valget av $\tau = -1$ fremfor for eksempel $\tau = 0$ er standard praksis: det gir en «ren» referanse som ikke selv er berørt av behandlingen, og gjør at $\beta_0$ viser den umiddelbare effekten i behandlingens første måned.

### Pre-trend-test

Pre-periodekoeffisientene $\beta_\tau$ for $\tau \leq -2$ bør under parallell-trendantagelsen ikke avvike systematisk fra null. Eventuelt positive pre-trend-koeffisienter ville indikere at de hardere rammede regionene allerede var i selvstendig oppgang — noe som er plausibelt gitt den observerte overutøvelsen og som i så fall ville gjøre post-behandlingsestimatene *konservative*.

Det gjennomføres en **felles Wald F-test** av $H_0: \beta_\tau = 0 \;\forall\; \tau \leq -2$, basert på cluster-robust kovariansmatrise. Fordi antall pre-periodekoeffisienter ($\approx 23$) overstiger $G - 1 = 11$ (den maksimale rangen til sandwich-estimatoren med 12 clustere), brukes pseudoinversmetoden (via eigenverdiregularisering) for å beregne en robust F-statistikk.

### Dynamisk effektbilde

Post-behandlingskoeffisientene $\beta_\tau$ for $\tau \geq 0$ kartlegger om effekten er umiddelbar og stabil, bygger seg opp gradvis (konsistent med at tiltak virker over tid), eller avtar (konsistent med tilpasning). Med kun 7 måneder post-behandling i tilgjengelige data er tolkningen av dynamikk begrenset, men mønsteret er likevel informativt.

---

## Inferens: wild cluster bootstrap

Med kun $G = 12$ clustere er asymptotiske clusterrobuste standardfeil upålitelige. Standardresultat fra asymptotisk teori forutsetter typisk $G \geq 30$–$50$ for god tilnærming, fordi sandwich-estimatoren konvergerer sakte når $G$ er liten. For å håndtere dette brukes **wild cluster bootstrap** (Cameron, Gelbach & Miller, 2008) med **Webb (6-punkt) vekter** (Webb, 2014).

### Hvorfor cluster-robust inferens i utgangspunktet?

Observasjonene innenfor samme cluster (region) er ikke uavhengige — de deler felles sjokk som ikke fanges opp av de faste effektene. Standard OLS-standardfeil undervurderer da usikkerheten. Cluster-robuste standardfeil løser dette ved å la residualene korrelere fritt innenfor cluster, men forutsetter uavhengighet *mellom* clustere.

### CR1-korreksjonen

Standardfeil beregnes med **CR1 korreksjon**:

$$\hat{V}_{\text{CR1}} = \frac{G}{G-1} \cdot \frac{N-1}{N-K} \cdot (X'X)^{-2} \sum_{g=1}^G \left(X_g' e_g\right)^2$$

Faktoren $\frac{G}{G-1} \cdot \frac{N-1}{N-K}$ er en small-sample-korreksjon. Uten den vil variansen systematisk underestimeres når $G$ og $N$ er moderate. CR1 er tilsvarende «HC1» for heteroskedastisitetsrobuste standardfeil, og er standard valg i paneldata-litteraturen.

### Problemet med få clustere

Selv med CR1 er konfidensintervallene basert på en $t$-fordeling med $G - 1 = 11$ frihetsgrader. Med få clustere er sandwich-estimatoren skjev nedover (undervurderer varians), og $t$-testens nullfordeling er ikke godt tilnærmet av en $t_{G-1}$-fordeling. Resultatet er for trange konfidensintervaller og for lave p-verdier — det vil si for hyppig forkasting av sanne nullhyposteser.

### Wild cluster bootstrap: prinsippet

I stedet for å stole på den asymptotiske fordelingen, simulerer wild cluster bootstrap den eksakte nullfordelingen til t-statistikken under $H_0: \beta = 0$ direkte fra dataene. Kjerneideén er å konstruere hypotetiske datasett som oppfyller nullhypotesen, men som bevarer clusterkorrelasjonsstrukturen ved å skalere residualene per cluster med tilfeldige vekter.

**Frisch–Waugh–Lovell (FWL)-projeksjon** utføres én gang før bootstrap-loopen: alle faste effekter partieres ut av $Y$ og $D$, slik at vi sitter igjen med de «within»-transformerte versjonene $\tilde{Y}$ og $\tilde{D}$. Dette er ekvivalent med å estimere på de faste-effekt-demeede dataene, og gjør loopen vesentlig raskere fordi hvert replikasjon bare krever en enkel skalarregresjon.

Under $H_0$ er $\tilde{Y}$ rent støy (nullresidual), og bootstrap-utfall konstrueres som $\tilde{Y}^*_b = w_g \cdot \tilde{Y}$, der $w_g$ er clusterspecifikke tilfeldige vekter.

### Webb-vektene

Webb (2014) viste at Rademacher-vektene $\{-1, +1\}$ kan gi dårlig dekning når $G$ er liten, fordi de kun har $2^G$ mulige realisasjoner av vektfordelingen (med $G = 12$ gir det bare 4096 unike bootstrap-datasett). Webb-vektene bruker i stedet en seks-punktfordeling:

$$w_g \in \left\{-\sqrt{\tfrac{3}{2}},\; -1,\; -\sqrt{\tfrac{1}{2}},\; \sqrt{\tfrac{1}{2}},\; 1,\; \sqrt{\tfrac{3}{2}}\right\}$$

Disse er valgt slik at $\mathbb{E}[w_g] = 0$, $\mathbb{E}[w_g^2] = 1$ og $\mathbb{E}[w_g^4] = \frac{3}{2}$ (matchen det fjerde momentet til normalfordelingen), noe som gir $6^G$ mulige realisasjoner og vesentlig bedre dekning enn Rademacher for $G < 20$.

### Algoritmen

1. **FWL-projeksjon**: beregn $\tilde{Y} = M_Z Y$ og $\tilde{D} = M_Z D$, der $M_Z$ er annihilatoren for alle faste effekter.
2. Beregn observert koeffisient $\hat{\beta} = (\tilde{D}'\tilde{D})^{-1}\tilde{D}'\tilde{Y}$ og t-statistikk $t_\text{obs} = \hat{\beta} / \widehat{\text{SE}}_{\text{CR1}}$.
3. For $b = 1, \ldots, 4\,999$:
   - Trekk $w_g \overset{\text{iid}}{\sim} \text{Webb}$ for $g = 1, \ldots, G$
   - $\tilde{Y}^*_b = w_g \cdot \tilde{Y}$ (broadcast per cluster — nullhypotesen innføres ved å bruke $\tilde{Y}$ direkte)
   - $\hat{\beta}^*_b = (\tilde{D}'\tilde{D})^{-1}\tilde{D}'\tilde{Y}^*_b$
   - Beregn $t^*_b = \hat{\beta}^*_b / \widehat{\text{SE}}^*_{\text{CR1},b}$
4. Bootstrap p-verdi: $\hat{p} = \frac{1}{B}\sum_{b=1}^B \mathbf{1}\{|t^*_b| \geq |t_\text{obs}|\}$

Bootstrap p-verdien er det **primære inferensresultatet**; asymptotiske p-verdier rapporteres som supplement.

---

## Referanser

- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414–427.
- Webb, M. D. (2014). *Reworking wild bootstrap based inference for clustered errors*. Queen's Economics Department Working Paper No. 1315.
