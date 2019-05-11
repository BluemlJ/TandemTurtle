Hi:) Ich hab mal zusammengefasst, was ich über die Notation rausgefunden hab:

### Bughouse Portable Game Notation (BPGN)

Das Format der Dateien in der  Datenbank (https://www.bughouse-db.org/) ist .bpgn.  Der Inhalt dieser Dateien sieht z.B. so aus:

{C:This is game number 3677475 at http://www.bughouse-db.org}
1A. e4{119.900} 1a. e6{119.410} 1B. e3{118.391} 2A. Nc3{119.800} 2a. Nc6{119.310} 1b. Nf6{119.553} 2B. d4{118.164} 3A. d4{119.700} 2b. e6{119.453} 3B. Nf3{117.807}3B. Nf3{117.807} 3b. d5{119.353} 3a. Bb4{117.875} 4B. Be2{117.419} 4A. Nf3{119.600} 4a.

...

 25a. Qxe8{85.142} 26A. P@f7{90.257} 26a. Qe7{83.132} 21B. P@f6{76.533} 27A. f8=Q+{88.500} 21b. R@f1+{93.172} 27a. B@e8{80.621} 22B. Rxf1{74.184} 28A. Qxe7+{87.509} 22b. gxf1=Q+{93.072} 28a. Nxe7{79.723} 23B. Kxf1{73.504} 29A. R@f8{85.212} 23b. P@g2+{88.214} 29a. Bd7{76.067} 24B. Ke2{71.106} 24b. B@f1+{87.227} 30A. N@f7+{81.403} 30a. Kc8{75.456} 25B. Ke1{69.279} 25b. Nxf3+{80.075} 26B. Qxf3{68.097} 26b. Q@e2+{79.975} 27B. Qxe2{59.377} 27b. Bxe2{79.875} 31A. Q@d8#{62.064}
{VaMPyReSLaYeR checkmated} 1-0

Soweit ich das erkennen konnte, ist die Bedeutung dieser Notation folgendermaßen:

Spieler A und Spieler b spielen gegen Spieler a und Spieler B.
Dabei spielen A und a am linken Brett und B und b am rechten.
A und B spielen mit weiß, a und b mit schwarz.

Die Figuren sind abgekürzt mit:

- K = King
- Q = Queen
- R = Rook
- B = Bishop
- N = kNight
- P = Pawn

Die Feld-Bezeichnung ist die Ziel-Position der Figur in diesem Zug.
In geschweiften Klammern ist der Zeitpunkt des Zugs.

Das heißt z.B. 3B. Nf3{117.807} bedeutet, dass Spieler B mit seinem dritten Zug zum Zeitpunkt 117.807 einen Springer (Knight) nach f3 setzt.

Ansonsten gibt es noch zu beachten:

- Das Figuren-Kürzel kann weggelassen werden, wenn es eindeutig ist, z.B. 1A. e4{119.900}
- Wenn es zwei Figuren gleicher Art gibt, die auf das gleiche Feld ziehen Können, dann wird auch das Ursprungs-Feld angegeben, z.B. 11a. Qe9e8{100.545}
- x bedeutet, dass bei dem Zug eine gegnerische Figur geschlagen wird, z.B. 22B. Rxf1{74.184}
- @ bedeutet, dass eine Figur gedropt wird, z.B. 26A. P@f7{90.257}
- g...=Q bedeutet, dass ein Bauer zur Dame wird (Promotion/Pappdame), z.B. 22b. gxf1=Q+{93.072}
- \+ bedeutet, dass durch den Zug der gegnerische König bedroht wird (Schach), z.B. 30A. N@f7+{81.403}
- \# bedeutet Schachmatt, z.B. 31A. Q@d8#{62.064}
