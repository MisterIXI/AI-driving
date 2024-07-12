# Todos
<!-- statt memorysize mehr channels

zwei getrennte Inputs: 1. Imagebuffer, 2. inputs (entweder 2x floats oder 2x onehot encoded vectoren für alle bins)
Output ist dann nur ein wert, die bewertung von state + action -->


<!-- - reward pro zeitscrhitt (immer mehr) und heftiger negativer bei Tod --> ist schon passiert
<!-- - vilt am anfang direkt aktion predicten ohne vollen memory buffer mit genullten screenshots (aka memory buffer füllen mit np.zeroes) -->
<!-- - bei _pull_batch_from_dataset bias für neuere batches einbauen. (wahrscheinlicher für neuere datensätze weils wahrshcienlich relevantere states sind) -->


### patrick
<!-- - Convolutions weglassen und direkt in denselayer reinpacken -->
- mehr daten pro "run", also zB 50x spielen und alles in eine h5 file packen, spielen ggf auch parallel

<!-- - statt (state + action => q_value der combo) soll sein: (state => q_values aller möglichen actionscombos) -->

## Todos 28.3.
- beim letzten frame (on death): future reward NICHT draufrechnen!!!!!!
- q_value clipping implementieren -> beim training die values vom target_model clippen damit die nicht wegrennen (wahrscheinlich erstmal nur das was auf "actual_value" draufgerechnet wird, sonst alle predictions vom target_model)
- gradient clipping implementieren und ausprobieren (maximum für gradient descent festlegen)
- testen ob die loss auch "nur auf daten" explodiert

- die ersten X runs vilt einfach kein training, oder weniger "traininsrunden" um erstmal daten zu sammeln
- wenn oben kram nicht geht: versuchen die loss und fit() zu masken damit nur die "echte" info fürs lernen genommen wird. zB: bei aktion [0,1,0] wird nur der q_value in der mitte genommen und die gegeben target_values sowie die berechnung vom model mit [0,1,0] multipliziert


- generell: mit in die daten aufnehmen wann epxloration statt gefunden hat um zu sehen was "echte" inputs sind und welche nicht


## für IPROF:
- github repo als ergebnis haben
- docu über lernprozess: zB was hat funktioniert, was wurde verändert
- docu: wenn man X verändert ergibt es das ergebnis Y