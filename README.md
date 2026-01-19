# Inleiding
Dit project bevat een computer vision–model gericht op het herkennen van visuele kenmerken die geassocieerd zijn met het Ehlers-Danlos Syndroom (EDS). EDS is een zeldzame erfelijke bindweefselaandoening die zich onder andere kan uiten in huid-, gewrichts- en weefselafwijkingen. Door de heterogeniteit van het syndroom is vroege en consistente herkenning vaak complex.

Het doel van dit project is om te onderzoeken in hoeverre moderne deep learning–technieken kunnen bijdragen aan ondersteunende patroonherkenning op beeldmateriaal dat relevant is voor EDS. Het model is ontwikkeld als research- en exploratieproject en is niet bedoeld als zelfstandig diagnostisch instrument.

De repository bevat:

- De gebruikte modelarchitectuur en trainingspipeline met OpenCV en Tensorflow in Python

- Voorverwerking van het beeldmateriaal

- Dataset met twee labels (EDS, niet EDS)

# Basis model keuze

MobileNetV2 is ontworpen met het oog op efficiëntie en robuustheid bij beperkte datasets. De architectuur maakt gebruik van depthwise separable convolutions en inverted residual blocks, wat resulteert in een relatief laag aantal parameters en een stabiel trainingsgedrag. Dit is met name relevant in een medische context zoals EDS, waar grote, goed gelabelde datasets vaak niet beschikbaar zijn.

Verder is MobileNetV2 gebruikt voor dit onderzoek: https://pmc.ncbi.nlm.nih.gov/articles/PMC12092329/. Dit onderzoek had veel succes met een 98.6% overall accuracy, waarbij huidaandoening werden herkend.


EfficientNet biedt uitstekende prestaties bij grootschalige datasets, maar is gevoeliger voor:

- overfitting bij kleinere datasets

- parameterafstemming

- variaties in beeldkwaliteit en resolutie

# ⁠Toekomstplan

Op dit moment is de trainings accuracy 86%, de daadwerkelijke accuracy is nog niet bekend, aangezien er te weinig data beschikbaar was. Verder zou het model verder getraind worden door bestaande images te spiegelen of om meer fragmenten te maken van EDS foto's op het web.

Op dit moment bestaat de dataset uit ongeveer 30 foto's per label, maar het zou enorm verbeterd kunnen worden. 

Verder is er nog geen user interface voor de gebruiker, daardoor is het model nog niet gebruiksvriendelijk.

# Toepassing

Tijdens ontwikkeling is er besloten om maar twee labels te gebruiken, waardoor het model eigenlijk pas ingezet kan worden bij het einde van het diagnoseproces, het model zou eerder kunnen worden toegepast, wanneer er meerdere labels zijn toegevoegd aan de dataset.
Op deze manier kan het model eerder worden toegepast in het proces (bijvoorbeeld bij de huisarts). 

