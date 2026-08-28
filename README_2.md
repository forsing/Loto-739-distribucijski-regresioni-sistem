⌘ + Shift + V


# Statistička analiza i istraživanje slučajnosti

## Cilj

Ovaj projekat istražuje da li istorijski podaci o izvlačenjima igre Loto 7 od 39 sadrže statističku strukturu, vremenske obrasce ili prediktivne informacije koje se mogu upotrebiti za izbor kombinacija sa očekivanim brojem pogodaka većim od slučajnog.

Cilj **nije** tvrdnja da se rezultati lutrije mogu predvideti sa sigurnošću.

Umesto toga, projekat posmatra Loto 7 od 39 kao statistički proces i ispituje da li istorijska zapažanja pružaju merljive prediktivne informacije o budućim izvlačenjima.

Glavno naučno pitanje jeste da li su uočeni istorijski obrasci u skladu sa uobičajenim slučajnim varijacijama konačnog uzorka ili sadrže ponovljiv signal koji se može primeniti na ranije neviđena izvlačenja.

---

## Istraživačko pitanje

Osnovno istraživačko pitanje glasi:

> Da li istorijska izvlačenja igre Loto 7 od 39 pružaju statistički pouzdane informacije o budućim izvlačenjima?

Prema pretpostavci nezavisnog slučajnog izvlačenja:

$$
P(X_t \mid X_{t-1}, X_{t-2}, \ldots) = P(X_t)
$$

Drugim rečima, poznavanje prethodnih izvlačenja ne bi trebalo da promeni stvarnu raspodelu verovatnoće narednog izvlačenja.

Projekat zato procenjuje da li istorijske informacije kao što su raspodela pojavljivanja, nedavnost, razmaci između pojavljivanja, survival i hazard raspodele, uslovni prelazi, odnosi parova, stabilnost, raznovrsnost i osobine mašinskog učenja mogu poboljšati prediktivne rezultate u odnosu na slučajni izbor sedam brojeva.

Analiza **ne pokušava** da dokaže slučajnost u apsolutnom smislu. Neuspeh u odbacivanju slučajnog modela nije isto što i dokaz da je proces savršeno slučajan.

---

## Podaci

Projekat koristi dva zasebna hronološka skupa podataka:

- Loto;
- Loto Plus.

Svaki red predstavlja jedno izvlačenje i sadrži tačno sedam različitih brojeva iz opsega 1–39.

Redosled podataka je:

```text
Prvi red  = najstarije izvlačenje
Poslednji red = najnovije izvlačenje
```

Loto i Loto Plus obrađuju se potpuno odvojeno.

Podaci jedne igre ne koriste se za obuku ili predikciju druge igre.

---

## Metodologija

Projekat koristi vremenski uređene istorijske podatke i strategije prvenstveno procenjuje pomoću walk-forward validacije.

Osnovna struktura eksperimenta je:

```text
Istorijski podaci
      |
      v
Period obuke
      |
      v
Pravljenje predikcije od sedam brojeva
      |
      v
Budući test period
      |
      v
Poređenje sa stvarnim izvlačenjem
```

Osnovna referentna vrednost jeste očekivani broj pogodaka pri ravnomernom izboru sedam brojeva iz opsega 1–39:

$$
E[H]
=
\frac{7 \times 7}{39}
=
\frac{49}{39}
=
1.256410
$$

Ovo predstavlja očekivani broj podudarnih brojeva po izvlačenju između dve nezavisne sedmočlane kombinacije iz opsega 1–39.

Projekat za glavne vremenski zavisne eksperimente izbegava uobičajenu slučajnu podelu na skupove za obuku i testiranje i umesto toga čuva hronološki redosled.

---

## Broj mogućih kombinacija

Ukupan broj različitih sedmočlanih kombinacija iz skupa od 39 brojeva iznosi:

$$
\binom{39}{7}
=
\frac{39!}{7!(39-7)!}
=
15\,380\,937
$$

Svaka dozvoljena kombinacija mora da sadrži:

- tačno sedam brojeva;
- sedam različitih brojeva;
- brojeve iz opsega 1–39;
- brojeve poređane rastućim redosledom.

---

## Slučajna raspodela broja pogodaka

Ako se jedna sedmočlana kombinacija poredi sa nezavisno izvučenom sedmočlanom kombinacijom, broj pogodaka prati hipergeometrijsku raspodelu:

$$
P(H=k)
=
\frac{
\binom{7}{k}
\binom{32}{7-k}
}{
\binom{39}{7}
}
$$

gde je:

- \(H\) broj pogodaka;
- \(k \in \{0,1,2,3,4,5,6,7\}\);
- 7 broj dobitnih brojeva;
- 32 broj brojeva koji nisu izvučeni;
- 39 ukupan broj mogućih brojeva.

Očekivani broj pogodaka je:

$$
E[H]
=
7 \cdot \frac{7}{39}
=
\frac{49}{39}
=
1.256410
$$

---

## Statistički testovi

Projekat može da sadrži više statističkih i računskih postupaka procene.

### Slučajna osnova

Svaka strategija poredi se sa teorijskim slučajnim očekivanjem:

$$
E[H]
=
\frac{49}{39}
=
1.256410
$$

Uočena razlika je:

$$
D
=
\bar H_{model}
-
\bar H_{random}
$$

Pozitivna razlika znači da je strategija tokom procenjivanog perioda ostvarila veći prosečan broj pogodaka od slučajne osnove.

Negativna razlika znači da je strategija ostvarila manji prosečan broj pogodaka od slučajne osnove.

### Monte Karlo simulacija slučajnog izbora

Monte Karlo simulacija može da napravi veliki broj slučajnih sedmočlanih kombinacija i proceni njihovu raspodelu pogodaka.

Simulacija može da se koristi za procenu:

- prosečnih slučajnih rezultata;
- donjih i gornjih percentilnih granica;
- empirijske verovatnoće ostvarivanja najmanje onakvog rezultata kakav je ostvario model;
- očekivane raspodele maksimalnog broja pogodaka;
- stabilnosti rezultata kroz različite vremenske periode.

Monte Karlo simulacija nije prediktivni model. Ona predstavlja način procene rezultata koji se mogu očekivati od slučajnog izbora.

### Hipergeometrijska slučajna osnova

Tačan hipergeometrijski model koristi se za broj pogodaka pri izboru sedam brojeva od ukupno 39 i poređenju sa sedam dobitnih brojeva.

Za razliku od Monte Karlo simulacije, hipergeometrijski model ne zavisi od slučajnog uzorkovanja.

### Bootstrap

Zamrznuta OOS validacija može ponovo da uzorkuje uočene razlike rezultata sa vraćanjem kako bi procenila bootstrap interval poverenja srednje razlike.

### Permutacioni test promene znaka

Zamrznuta validacija može da napravi nultu raspodelu nasumičnim menjanjem znaka uočenih razlika i da uporedi dobijene srednje razlike sa uočenom srednjom razlikom.

### Važno ograničenje

Statistički testovi koji nisu stvarno implementirani i izvršeni ne smeju se predstavljati kao sprovedeni.

To se odnosi na:

- Hi-kvadrat testove saglasnosti;
- testove autokorelacije;
- testove nizova;
- Ljung–Box testove;
- formalne testove entropije;
- testove nezavisnosti Markovljevih lanaca;
- Bajesovo zaključivanje.

---

## Analiza istorijskih obrazaca

### Osnovna stopa pojavljivanja

Za svaki broj od 1 do 39, teorijska verovatnoća pojavljivanja u jednom izvlačenju iznosi:

$$
p_0
=
\frac{7}{39}
=
0.179487
$$

Očekivani broj pojavljivanja određenog broja u \(T\) izvlačenja iznosi:

$$
E[C_i]
=
T \cdot \frac{7}{39}
$$

Istorijska stopa pojavljivanja broja \(i\) može se zapisati kao:

$$
f_i
=
\frac{c_i}{T}
$$

gde je:

- \(c_i\) broj istorijskih pojavljivanja broja \(i\);
- \(T\) broj izvlačenja.

Odstupanje od osnovne stope može se zapisati kao:

$$
Odstupanje_i
=
f_i
-
\frac{7}{39}
$$

Relativna snaga može se izračunati kao:

$$
Snaga_i
=
\frac{
f_i
}{
7/39
}
$$

Sama istorijska stopa pojavljivanja ne predstavlja dokaz prediktivne prednosti.

### Nedavnost

Nedavne stope mogu se računati u vremenskim prozorima različitih dužina.

Na primer:

$$
Stopa_{20}
=
\frac{
BrojPojavljivanja_{20}
}{
20
}
$$

Razlika između kratkoročne i dugoročnije stope može se zapisati kao:

$$
Momentum
=
Stopa_{20}
-
Stopa_{50}
$$

Korisnost nedavnosti mora se dokazati na hronološki kasnijim, ranije neviđenim izvlačenjima.

### Razmak između pojavljivanja

Razmak broja predstavlja broj izvlačenja od njegovog prethodnog pojavljivanja.

Za broj \(i\) u trenutku \(t\):

$$
Gap_{i,t}
=
t
-
LastSeen_{i,t}
$$

Veliki razmak ne znači da broj matematički „mora” uskoro da se pojavi.

Razmak se koristi kao ulazna distribucijska osobina čija korisnost mora biti potvrđena van uzorka.

### Empirijska raspodela razmaka

Za završene istorijske razmake broja mogu se računati:

- 10. percentil;
- 25. percentil;
- medijana;
- 75. percentil;
- 90. percentil;
- empirijska kumulativna raspodela;
- empirijska survival raspodela;
- hazard;
- entropija raspodele.

Empirijska kumulativna raspodela za trenutni razmak \(g\) može se zapisati kao:

$$
F(g)
=
P(G \le g)
$$

Empirijska survival funkcija može se zapisati kao:

$$
S(g)
=
P(G \ge g)
$$

Empirijski hazard može se približno zapisati kao:

$$
h(g)
=
\frac{
BrojDogađajaNaRazmaku(g)+1
}{
BrojRazmakaPodRizikom(g)+2
}
$$

Dodavanje jedinice i dvojke predstavlja zaglađivanje kojim se izbegava nestabilnost pri malom broju zapažanja.

### Kontinuirana time-to-event meta

Za svaki broj koristi se kontinuirana meta izvedena iz broja izvlačenja do narednog pojavljivanja:

$$
Y_{i,t}
=
\log
\left(
\frac{
39/7
}{
Wait_{i,t}
}
\right)
$$

gde je:

- \(Wait_{i,t}\) broj izvlačenja do narednog pojavljivanja broja \(i\);
- \(39/7\) osnovno očekivano vreme čekanja.

Veća vrednost mete odgovara skorijem budućem pojavljivanju.

Ova meta je kontinuirana i ne predstavlja klasifikacionu oznaku.

### Survival i hazard raspodela

Survival i hazard osobine koriste se kao opis istorijske raspodele čekanja.

One ne dokazuju da je broj „dužan” da se pojavi.

Njihova prediktivna vrednost mora se meriti kroz walk-forward validaciju.

### Uslovni prelazi

Za broj \(a\) iz prethodnog izvlačenja i kandidata \(b\), uslovna verovatnoća može se proceniti kao:

$$
P(b_t \mid a_{t-1})
=
\frac{
Count(a_{t-1},b_t)
+
\alpha p_0
}{
Count(a_{t-1})
+
\alpha
}
$$

gde je:

$$
p_0
=
\frac{7}{39}
$$

a \(\alpha\) jačina prethodne raspodele.

Odstupanje uslovne verovatnoće od osnovne stope iznosi:

$$
TransitionDeviation
=
P(b_t \mid a_{t-1})
-
\frac{7}{39}
$$

### Analiza parova

U jednoj sedmočlanoj kombinaciji postoji:

$$
\binom{7}{2}
=
21
$$

različit par.

Ukupan broj mogućih parova među 39 brojeva iznosi:

$$
\binom{39}{2}
=
741
$$

Osnovna verovatnoća pojavljivanja određenog para u jednom izvlačenju iznosi:

$$
p_{pair}
=
\frac{
\binom{7}{2}
}{
\binom{39}{2}
}
=
\frac{21}{741}
=
0.028340
$$

Očekivano vreme čekanja na određeni par iznosi:

$$
E[Wait_{pair}]
=
\frac{1}{p_{pair}}
=
\frac{741}{21}
=
35.285714
$$

Kontinuirana meta para može se zapisati kao:

$$
Y_{pair,t}
=
\log
\left(
\frac{
741/21
}{
Wait_{pair,t}
}
\right)
$$

### Ko-pojavljivanje

Istorijsko ko-pojavljivanje dva broja procenjuje se samo iz podataka dostupnih pre ciljnog izvlačenja.

Uslovna procena može se zapisati kao:

$$
P(j \mid i)
=
\frac{
Count(i,j)
+
\alpha p_0
}{
Count(i)
+
\alpha
}
$$

Odstupanje od osnovne stope iznosi:

$$
CooccurrenceDeviation
=
P(j \mid i)
-
\frac{7}{39}
$$

### Uzastopni brojevi

Za kombinaciju se može računati broj susednih parova čija je razlika jednaka jedan.

Uzastopni brojevi posmatraju se kao strukturna osobina kombinacije, a ne kao samostalan dokaz prediktivnosti.

### Neparni i parni brojevi

Za svaku kombinaciju može se računati:

- broj neparnih brojeva;
- broj parnih brojeva;
- odnos neparnih i parnih brojeva.

Ove veličine predstavljaju strukturne osobine.

### Niski i visoki brojevi

Brojevi se mogu podeliti na:

- niske brojeve: 1–20;
- visoke brojeve: 21–39.

Broj niskih i visokih brojeva koristi se kao strukturna osobina kombinacije.

### Zbir, prosek, standardna devijacija i raspon

Strukturne osobine kombinacije mogu da obuhvate:

- zbir;
- prosek;
- standardnu devijaciju;
- najmanji broj;
- najveći broj;
- raspon;
- broj uzastopnih parova.

Ove osobine ne predstavljaju zasebne statističke testove.

---

## Prediktivni modeli

### Kontinuirani regresor brojeva

Istorijski podaci pretvaraju se u primere na nivou brojeva.

Za svaki broj od 1 do 39 prave se osobine dostupne pre ciljnog izvlačenja.

Regresor predviđa kontinuirani distribucijski skor:

$$
\hat Y_{i,t}
=
f(X_{i,t})
$$

gde je:

- \(X_{i,t}\) vektor istorijskih osobina broja;
- \(\hat Y_{i,t}\) predviđeni kontinuirani skor.

Model ne daje klasifikacionu oznaku, već kontinuirani skor za rangiranje.

### Brzi kombinacijski regresor

Brzi regresor koristi kompaktan skup osobina kombinacije.

Njegova namena je da paketno oceni svih:

$$
15\,380\,937
$$

dozvoljenih kombinacija.

Kombinacije se ne čuvaju sve istovremeno u memoriji, već se obrađuju u paketima.

Posle svakog paketa zadržava se samo unapred određen broj najbolje rangiranih kandidata.

### Precizni kombinacijski regresor

Precizni regresor koristi širi skup osobina:

- sedam sortiranih pojedinačnih skorova;
- svih 21 sortiranih skorova parova;
- prosek i standardnu devijaciju pojedinačnih skorova;
- prosek i standardnu devijaciju parnih skorova;
- strukturne osobine kompletne kombinacije.

Precizni regresor ocenjuje najbolje kandidate koje je zadržala prva faza.

Precizna faza koristi se samo ako na odvojenoj hronološkoj validaciji ostvari:

- manji MAE od brze faze;
- najmanje jednak recall najbolje ciljne kombinacije.

Ako precizna faza ne zadovolji oba uslova, konačan izbor pravi brza faza.

---

## Kontinuirana kombinacijska meta

Kombinacijska meta objedinjuje tri komponente:

1. pojedinačne time-to-event skorove;
2. parne time-to-event skorove;
3. vremenski ponderisan budući presek.

Meta se može zapisati kao:

$$
Y_{combo}
=
w_n Y_{number}
+
w_p Y_{pair}
+
w_o Y_{overlap}
$$

uz:

$$
w_n + w_p + w_o = 1
$$

Primer težina:

$$
w_n = 0.50
$$

$$
w_p = 0.30
$$

$$
w_o = 0.20
$$

Pojedinačna komponenta predstavlja prosek kontinuiranih meta sedam izabranih brojeva:

$$
Y_{number}
=
\frac{1}{7}
\sum_{i \in C}
Y_{i,t}
$$

Parna komponenta predstavlja prosek kontinuiranih meta svih 21 parova:

$$
Y_{pair}
=
\frac{1}{21}
\sum_{(i,j)\in C}
Y_{ij,t}
$$

Vremenski ponderisan presek koristi više budućih izvlačenja sa opadajućim težinama:

$$
Y_{overlap}
=
\frac{
\sum_{h=0}^{H-1}
\lambda^h
\left(
\frac{
|C \cap X_{t+h}|
}{7}
-
\frac{7}{39}
\right)
}{
\sum_{h=0}^{H-1}
\lambda^h
}
$$

gde je:

- \(C\) kandidat kombinacija;
- \(X_{t+h}\) buduće izvlačenje;
- \(H\) broj budućih izvlačenja;
- \(\lambda\) faktor vremenskog opadanja.

---

## Near-miss kombinacije

Za svako stvarno istorijsko izvlačenje prave se kombinacije koje se od njega razlikuju za tačno jedan broj.

Ako je stvarna kombinacija:

$$
C
=
\{c_1,c_2,\ldots,c_7\}
$$

near-miss kombinacija dobija se zamenom jednog broja:

$$
C'
=
C
-
\{c_i\}
+
\{r\}
$$

gde \(r\) nije član stvarne kombinacije.

Ovi primeri pomažu modelu da razlikuje veoma slične kombinacije.

---

## Istorijski pronađeni kandidati

Obuka preciznog regresora ne koristi samo stvarna izvlačenja i near-miss kombinacije.

U svakom istorijskom preseku prva faza pravi i širi skup jakih kandidata na osnovu tada dostupnih skorova.

Time se smanjuje razlika između:

- kombinacija koje model vidi tokom obuke;
- kombinacija koje dobija tokom konačnog prerangiranja.

Sve osobine istorijskih kandidata moraju biti napravljene isključivo iz podataka dostupnih pre ciljnog izvlačenja.

---

## Walk-forward validacija

Walk-forward validacija čuva hronološki redosled.

Osnovna struktura je:

```text
Obuka: ranija istorijska izvlačenja
Test:  kasnija, ranije neviđena izvlačenja
```

Model u trenutku \(t\) sme da koristi samo informacije dostupne pre trenutka \(t\).

Ciljno izvlačenje dodaje se istorijskom stanju tek nakon pravljenja njegovih ulaznih osobina.

---

## Grupna hronološka validacija

Sve kombinacije napravljene iz istog istorijskog izvlačenja pripadaju istoj grupi.

Podela na obuku i validaciju vrši se po celim grupama:

```text
Ranije grupe izvlačenja  → obuka
Kasnije grupe izvlačenja → validacija
```

Kombinacije nastale iz istog izvlačenja ne smeju biti podeljene između obuke i validacije.

Ovo sprečava da veoma slične kombinacije istog porekla dospeju u oba skupa.

---

## Recall prve faze

Prva faza mora da zadrži dovoljno širok skup kandidata da dobra kombinacija ne bude odbačena pre preciznog rangiranja.

Recall prve faze može se zapisati kao:

$$
Recall@K
=
\frac{
BrojGrupaUGdeJeNajboljiCiljniKandidatUTopK
}{
UkupanBrojValidacionihGrupa
}
$$

Veći recall znači da prva faza češće zadržava najbolji ciljni kandidat.

Recall se meri samo na hronološki kasnijim validacionim grupama.

---

## MAE

Srednja apsolutna greška kontinuiranog regresora iznosi:

$$
MAE
=
\frac{1}{N}
\sum_{i=1}^{N}
|
Y_i
-
\hat Y_i
|
$$

Manji MAE predstavlja preciznije predviđanje kontinuirane mete.

MAE sam po sebi nije dovoljan.

Model mora da pokaže i zadovoljavajuću sposobnost rangiranja najboljih kombinacija.

---

## Stabilno rangiranje

Kada dve kombinacije imaju isti predviđeni skor, koristi se stabilan redosled.

Stabilno rangiranje omogućava:

- ponovljiv rezultat;
- isti rezultat pri istom ulazu;
- izbegavanje proizvoljnog menjanja kandidata sa jednakim skorom.

Nasumično razrešavanje izjednačenih skorova ne koristi se za konačnu NEXT predikciju.

---

## Paketna obrada svih kombinacija

Svih:

$$
15\,380\,937
$$

kombinacija obrađuje se u paketima.

Primer:

```text
Generiši paket kombinacija
        ↓
Napravi kompaktne osobine
        ↓
Predvidi kontinuirane skorove
        ↓
Zadrži samo najbolje kandidate
        ↓
Pređi na sledeći paket
```

Ovaj pristup sprečava držanje svih kombinacija i svih njihovih osobina u memoriji istovremeno.

Na kraju prve faze preciznom regresoru prosleđuje se samo unapred određen broj najboljih kandidata.

---

## Odvojena obrada Lota i Lota Plus

Loto i Loto Plus prolaze kroz isti postupak, ali odvojeno.

Za svaku igru zasebno se prave:

- istorijsko distribucijsko stanje;
- razmaci;
- survival i hazard osobine;
- uslovni prelazi;
- odnosi parova;
- podaci za obuku;
- hronološke grupe;
- validacione metrike;
- konačna NEXT predikcija.



---

## Zamrznuta validacija van uzorka

Nakon izbora konačne metodologije potrebno je zamrznuti:

- osobine;
- ciljne funkcije;
- težine komponenti;
- parametre modela;
- veličinu prve faze;
- broj kandidata za precizno rangiranje;
- pravilo izbora završne faze.

Posle zamrzavanja ne sme se menjati model na osnovu rezultata novih izvlačenja.

Pravi OOS postupak je:

```text
Zamrznuti metodologiju
        ↓
Napraviti NEXT predikciju
        ↓
Sačekati stvarno novo izvlačenje
        ↓
Zabeležiti rezultat bez promene modela
        ↓
Nastaviti prikupljanje OOS rezultata
```

---

## Preprilagođavanje i pretraživanje podataka

Projekat koristi mere zaštite od neposrednog gledanja unapred:

- hronološku obuku i testiranje;
- walk-forward obradu;
- istorijske osobine napravljene pre ciljnog izvlačenja;
- grupnu hronološku validaciju;
- kontinuirane mete razrešene pre granice obuke;
- odvojenu obradu Lota i Lota Plus.

Ipak, iterativni razvoj može da stvori rizik izbora modela.

Testiranje velikog broja:

- osobina;
- ciljnih funkcija;
- težina;
- prozora;
- veličina kandidatnog skupa;
- parametara modela;

povećava verovatnoću da najbolji istorijski rezultat nastane slučajno.

Zato konačan model mora biti zamrznut i procenjen na stvarno novim izvlačenjima.

---

## Ograničenja

### Slučajnost

Loto izvlačenja su probabilistički događaji.

Istorijska odstupanja od očekivanih raspodela mogu se pojaviti prirodno.

### Konačna veličina uzorka

Čak ni nekoliko hiljada izvlačenja ne garantuje da mala odstupanja predstavljaju stvarne strukturne efekte.

### Veliki prostor kombinacija

Pregled svih 15.380.937 kombinacija garantuje da je pronađena najbolje ocenjena kombinacija prema datom modelu.

To ne garantuje da je model naučio stvarnu prediktivnu vezu.

### Višestruka poređenja

Testiranje velikog broja modela i postavki povećava verovatnoću slučajnog pronalaženja prividno dobrog istorijskog rezultata.

### Preprilagođavanje

Regresor može naučiti istorijski šum bez učenja veze koja se može primeniti na buduća izvlačenja.

### Pristrasnost izbora

Izbor postavke sa najboljim istorijskim rezultatima uvodi pristrasnost ako se rezultat zatim predstavlja kao nezavisna potvrda.

### Neizvesnost predikcije

Veći prosečan broj pogodaka nije isto što i veća verovatnoća pogađanja svih sedam brojeva u istoj kombinaciji.

### Nepotpuno formalno testiranje slučajnosti

Zaključci o slučajnosti moraju biti ograničeni na testove koji su stvarno implementirani i izvršeni.

---

## Tumačenje rezultata

Rezultati validacije treba da prikažu najmanje:

- broj hronoloških grupa za obuku;
- broj odvojenih validacionih grupa;
- MAE brze faze;
- MAE precizne faze;
- recall brze faze;
- recall precizne faze;
- izabranu završnu fazu;
- broj pregledanih kombinacija.

Primer strukture ispisa:

```text
NEXT: xx, xx, xx, xx, xx, xx, xx
CSV redova: ...
Pregledano kombinacija: 15,380,937
Hronoloških grupa za obuku: ...
Odvojenih validacionih grupa: ...
Fast MAE: ...
Precise MAE: ...
Fast recall: ...
Precise recall: ...
Izabrana završna faza: ...
```

Precizna faza treba da bude izabrana samo kada istovremeno:

1. ostvaruje manji validacioni MAE;
2. ostvaruje najmanje jednak validacioni recall.

U suprotnom se koristi rezultat brze faze.

---

## Zaključak

Projekat ne dokazuje da je Loto 7 od 39 matematički predvidiv niti da istorijska izvlačenja garantuju prednost.

Projekat pravi metodološki kontrolisan sistem koji:

- koristi samo hronološki ranije podatke;
- pravi kontinuirane distribucijske mete;
- obrađuje pojedinačne i parne odnose;
- koristi hard-negative kombinacije;
- koristi grupnu vremensku validaciju;
- meri recall prve faze;
- pregleda svih 15.380.937 kombinacija;
- daje jednu završnu NEXT kombinaciju za Loto;
- daje jednu završnu NEXT kombinaciju za Loto Plus.

Pregled svih mogućih kombinacija garantuje samo da je izabrana kombinacija sa najvećim skorom prema obučenom modelu.

To nije garancija da će ta kombinacija biti izvučena.

Najvažniji naučni test ostaje zamrznuta validacija na stvarno budućim, ranije neviđenim izvlačenjima.

Nijednu pojedinačnu kombinaciju ne treba tumačiti kao garantovano dobitnu.

Svrha projekta jeste statističko istraživanje predvidljivosti, zavisnosti, distribucijskih obrazaca i mogućeg signala — a ne garantovano predviđanje dobitnih brojeva.

