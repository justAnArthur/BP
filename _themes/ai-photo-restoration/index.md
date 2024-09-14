# Exersice

## Original

Reštaurovanie obrazov je úloha počítačového videnia, ktorá zahŕňa vrátenie poškodených obrazov do pôvodného stavu. Umelá inteligencia, konkrétne neurónové siete v poslednom čase priniesli revolúciu v tejto oblasti tým, že poskytujú moderné výsledky pri rôznych úlohách obnovy, napríklad denoizáciu obrazu založenú na hlbokom učení alebo vysokom rozlíšení. Tieto metódy majú schopnosť, ktorá sa dá využiť v mnohých aplikáciách s tým, že sú vytrénované na súbore údajov alebo bez nich.

Analyzujte súčasný stav reštaurovania obrazov (angl. photo restoration) smerom k praktickému používaniu a v kontexte umelej inteligencie. Navrhnite a implementujte základný modul ako jadro inteligentnej aplikácie na spracovanie a obnovenie obrazov. Ako praktickú ukážku vývoja softvéru, vytvorte docker kontajner pre Vašu aplikáciu. Zvoľte vhodné metriky na meranie výsledkov a ich vyhodnoťte.

## English

Image restoration is a computer vision task that involves returning _damaged images_ to their original state. Artificial intelligence, specifically neural networks, has recently revolutionized this field by providing modern results in various recovery tasks, such as image denoization based on deep learning or high resolution. These methods have a capability that can be used in many applications, with or without training on a dataset.

>**Image Denoising** is a computer vision task that involves removing noise from an image

Analyze the current state of photo restoration towards practical use and in the context of artificial intelligence. Design and implement the base module as the core of a smart image processing and recovery application. As a practical demonstration of software development, create a docker container for your application. Choose appropriate metrics to measure results and evaluate them.

# Questions

1. What kind of images the model should be able to restore?
   - black-and-white photos, damaged prints, or ancient.
2. Appropriate metrics ?

# Problems

- noise, scratches, and other imperfections.
- Colorization
- Resolution Enhancement
- Face restoring