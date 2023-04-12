# Opis problema

Problem trgovačkog putnika predstavlja jedan od najpoznatijih problema kombinatorne optimizacije. Problem trgovačkog putnika možemo da formulišemo na sledeći način: Od svih Hamiltonovih puteva odabrati onaj koji je prema zadatom kriterijumu **optimalan** (kriterijum optimalnosti može npr. biti dužina ili cena puta).

**Definicija** : Hamiltonov put je put koji prolazi kroz sve čvorove grafa TAČNO JEDANPUT. Put koji se završava u istom čvoru a kroz sve ostale čvorove prolazi tačno jedanput naziva se zatvoren Hamiltonov put ( Hamiltonov put formira konturu koja sadrži sve čvorove grafa pa se umesto pojma Hamiltonov put češće korsiti pojam Hamiltonove konture). Graf koji ima Hamiltonovu konturu se naziva Hamiltonovim grafom.



# Uvod

U ovoj implementaciji, rešenje problema trgovačkog putnika aproksimiramo korišćenjem genetskog algoritma.Genetski algoritam pripada grupi evolutivnih algoritama koji su inspirisani procesom evolucije bioloških sistema.

Kriterijum optimalnosti predstavlja dužinu puta, a optimalno rešenje je ona jedinka čija je putanja najkraća.

# Implementacija

Na samom početku je generisana početna populacija, koja se sastoji od korisnički zadanog broja jedinki, dok svaka jedinka predstavlja moguće rešenje problema,tj. predstavlja zatvoreni Hamiltonov put koji kreće iz zadatog grada. Svaka jedinka se sastoji od gena, koji u ovom slučaju predstavljaju gradove.

Izračunava se prilagođenost svake jedinke, nakon čega se vrši selekcija jedinki za ukrštavanje. Jedinke su sortirane u opadajućem redosledu po dužini i svakoj je dodeljena kumulativna suma dužine puta. Kumulativna suma dužine puta predstavlja sumu dužine puta te jedinke kao I svih jedinki čija je dužina veća od dužine te jedinke. Pored kumulativne sume, jedinkama je dodeljena I verovatnoća odabira koja se računa tako što kumulativnu dužinu puta jedinke delimo sumom dužina svih jedinki u populaciji.

Ako je u dozvoljen elitizam, određeni broj jedinki, onih sa najkraćom putanjom, se automatski selektuje za ukrštanje i prenosi u sledeću generaciju bez mutacije.

Ukrštanje se vrši nasumičnim odabirom niza nasumične dužine uzastopnih gena jednog roditelja,a ostatak gena se uzima od drugog roditelja, ali tako da ne dođe do ponavljanja gena,sem prvog i poslednjeg(prvi gen mora biti isti kao i poslednji). Ta dva dela se ukrštaju i prenose u sledeću generaciju.

Mutacija predstavlja nasumičnu zamenu određena dva gena. Verovatnoća da će doći do mutacije predstavlja broj od 0 do 1 i unosi se kao parametar funkcije.

Pri pozivu algoritma unose se ovi parametri:
***geneticAlgortihm(starting_point,mutationRate,elitismRate,generation_num,population_size)***

**starting\_point** – indeks grada iz kog se kreće

**mutationRate** – verovatnoća da dođe do mutacije određenog gena

**elitismRate** – stepen elitizma izražen brojevima od 0 do 1 (0 predstavlja 0%, a 1 predstavlja 100% populacije)

**generation\_num** – broj željenih generacija, najbolja jedinka iz ove generacije predstavlja aproksimaciju rešenja problema

**population\_size –** broj jedinki u svakoj populaciji

# Zaključak

Algoritam se pokazao kao efektivan. U proseku, algoritam već posle 200 generacija nalazi duplo bolja rešenja od početnih. Ovaj algoritam predstavalja samo jedan od algoritama koji aproksimira rešenje problema trgovačkog putnika. Prednost ovog algoritma je relativno jednostavna implementacija.

# Primer izlaza

![](/home/micax/Pictures/Figure_1.png)