# Génération d'image niveau débutant: guide synthétique par Coco

Je fais ce post pour résumer les quelques bases j’ai pu trouver par ci par là en faisant mes recherches sur la génération d’images par IA. Le but étant de donner des bases pour ceux qui veulent découvrir un peu la génération d’images. Ici : https://github.com/altryne/awesome-ai-art-image-synthesis vous avez un très bon hub pour trouver plein de ressources sur le sujet.

## Les Modèles Existants

Aujourd’hui il y a 3 modèles IA disponibles : **DALLE-E 2** (Closed-source payant), **MidJourney** (Closed-source payant), **Stable Diffusion** (Open-Source Gratuit)

### Les hébergeurs et les prix

**Payant et propriétaires :**  
- _DALL-E2:_ Waiting list — Gratuit : 50 credit + 15/mois—Payant: 15€/115 credit https://openai.com/dall-e-2/
- _MidJourney :_ Gratuit 25 crédits + 0/mois — Payant : 10 €/200 crédits par mois, 30 €/15 000 crédits par mois https://www.midjourney.com/home/  

**Payant, reposant sur Stable Diff (open-source):**
- _DreamStudio:_ Gratuit : 200 credit + 0/mois — Payant : 2 €/200 credit, 10€/1000 credit https://beta.dreamstudio.ai
- _NightCafe:_ Gratuit : 5 crédits par jour — Payant : 10 €/100 credit, 20€/250 credit https://creator.nightcafe.studio/  

**Gratuit, Stable Diffusion à autohéberger**

- **StableDiffusion WEBUI** : Un projet d’interface web pour l’utilisation du modèle gratuit Stable Diffusion. Ce code est à télécharger et à exécuter sur sa propre machine (soit à la maison si vous avec une carte graphique, soit sur un serveur qui en possède une) https://github.com/sd-webui/stable-diffusion-webui

- **Google Colab** : la plateforme de Google pour faire tourner du Python avec leur GPU. Si vous n’avez pas de GPU ou de serveur, vous pouvez utiliser cette plateforme pour faire tourner facilement StableDiffusion WEBUI https://github.com/altryne/sd-webui-colab. Prix : offre gratuite + offre 11 €/mois + offre 45 €/mois. À partir du 31 septembre, le décompte des ressources disponibles pour chaque plan sera clairement affiché.

- **Mon propre hébergement de labo** : si vous voulez expérimenter, je peux faire tourner moi-même (quand je suis présent) une instance de StableDiffusion sur le serveur de mon labo. L’instance peut tourner plusieurs heures tant que je suis là, il suffit de me demander, mais elle s’éteint quand je suis déconnecté.

### Les tâches que les IA peuvent effectuer

- *Text2Img* : on entre une phrase à partir de laquelle on génère une image, c’est le mode de base.  
- *Img2Img : In/OutPainting*: à partir d’une image, on peut masquer certaines zones et ajouter une phrase pour regénérer l’espace. Sert par exemple pour « corriger » ou « améliorer » une image ou même étendre une image avec plus de contenu. On peut aussi faire la fusion entre deux images en « peignant » entre deux images, en replissant un « trou ».  
- *Upscaling* : comme les images sont souvent générée avec une faible résolution (512x512), il existent des modèles capables d’augmenter la résolution de nos images automatiquement  

## Focus sur Stable Diffusion et ses paramètres

Je focus sur stable diffusion (et en particulier autohébergée), mais la partie « prompt » _(phrase de génération)_ fonctionnera aussi pour DALL-E2, MidJourney, DreamStudio et Nighcafe. Le reste est moins pertinent car comme vous êtes limités en génération sur les services payant, il n'y a pas beaucoup de place pour le trial-and-error et l'exploration de seed, sampler etc.

### Le paramètre CFG

Le paramètre CFG (Classifier Free Guidance) est un paramètre numérique qui indique à l’IA à quel point elle doit suivre votre prompt. Souvent les gens restent avec une valeur de 7 à 11. Plus le prompt est qualitatif, plus le CFG peut être augmenté pour forcer l’IA à le suivre à la lettre. Voici l’échelle des valeurs :

- 2 - 6 : l’IA est libre
- **7 - 11** : l’IA collabore et suit le prompt
- 12 -15 : l’IA est forcée de suivre le prompt correctement
- 16 - 20 : l’IA suit totalement le prompt sans liberté

### Les Samplers

_N. B. les infos ici sont expérimentales et apparemment beaucoup vont changer avec la v1.5 de SD, les variations semblent très faibles entre les samplers en v1.5._  
Les samplers sont différents « moteurs » qui vont générer vos images. Ils ont chacun leur propres caractéristique et style, leur choix est important (en SD v1.4 en tout cas.).

- **k_lms** (défaut) : la valeur sure. 50 step, CFG 7-8, il est plutôt rapide et donne de bons résultats
- **DDIM** : pour itérer très rapidement. Avec 8 à 15 steps, il fonctionne très vite et permet de générer plein d’images pour un prompt donné sur plusieurs seed afin de trouver la seed parfait et de vérifier si un concept de prompt peut fonctionner. _Style : Plutôt réaliste._
- k*euler_a : comme DDIM ultra variable entre les steps, les images changent beaucoup. \_Style : bon pour la « fantasy »*
- **k_dpm_2_a** : Très bon et très lent. 30—80 steps voir + une fois qu’on a une bonne seed et/ou un bon prompt pour faire une belle image (100-150 steps possiblement).
- PLMS : demande beaucoup de travail/générations _Style : pour les petits détails_  

**Update 25/09: nouvelle étude.** https://www.reddit.com/r/StableDiffusion/comments/xmwcrx/a_comparison_between_8_samplers_for_5_different/
- 8 à 30 steps pour itérer vite, 100 steps pour l'image finale (convergence de tout les samplers).
- K_LMS, K_HEUN and K_DPM_2 dans la plus part des cas. (Attention, à 8 steps, K_HEUN et K_DPM_2 ne sont pas recommandés)
- K_EULER_A pour de la variabilité forte

### La Seed

#### Trouver une bonne seed

**Trouver une bonne seed est crucial pour générer une bonne image**. La seed, c’est un nombre aléatoire utilisé par l’IA pour générer une image. Pour un même prompt, deux résultats pourront être très différent niveau qualité juste à cause d’une mauvaise seed. Donc une fois qu’on a un prompt qu’on estime un minimum solide, il est intéressant de faire du « seed hunting ». C’est à dire, générer plein d’images pour un même prompt, possiblement avec un nombre de steps faible (et sampler K_LMS/DDIM), à chaque fois avec une seed différente et garder la seed d’un résultat qui est satisfaisant.

#### Fixer une Seed

Autant trouver la bonne seed est cruciale, autant fixer une seed (la définir et ne pas la changer) peut être très utile pour une chose : tester son prompt. En effet, en fixant une seed, on peut modifier notre prompt et voir directement l’effet de chaque mot, chaque ajout, chaque suppression sur notre résultat. Par exemple : ajouter/supprimer des artistes.  
_NB : Augmenter le CFG peut être utile pour forcer l’IA à suivre les modifs de prompt._

## Focus sur les Prompt

Le prompt c’est la phrase que vous donnez à l’IA pour générer l’image. Développer un bon prompt est donc essentiel, ça passe par beaucoup de « trial and errors » et c’est ce qu’on appelle le « prompt engineering ». Ci-dessous pleins de conseils pour créer un bon prompt et trouver des idées.

### Structure conseillée

On conseille une structure en quatre parties :

1. **Le sujet** principal : « Pandas », « Red-haired woman », « A warrior with a sword »
2. **Le Style** : « Realistic », « Oil painting », « Pencil drawing », « Concept art ». Pour un style réaliste : « hyperrealistic », « realistic », « a photo of \<prompt\> ‘a photograph of \<prompt\>’
3. **Un Artiste** : “by Picasso” ou juste “Picasso”
4. **Touches finales** : on peut à la fin ajouter un nombre important de modificateurs plus ou moins spécifique, du genre “Highly detailed, surrealism, trending on art station, triadic color scheme, smooth, sharp focus, matte, elegant, the most beautiful image ever seen, illustration, digital paint, dark, gloomy, octane render, 8k, 4k, washed colors, sharp, dramatic lighting, beautiful, post-processing, picture of the day, ambient lighting, epic composition”

Exemple de prompt : _“Red-haired woman, oil painting, Picasso, dark, highly detailed”_

### Trouver de l’inspiration et les mots-clés

**La bible pour trouver l’inspiration** : un site qui permet de taper des mots-clés, de voir des images déjà générées avec SD et le prompt associé : https://lexica.art/  
Un code pour envoyer une image et obtenir les mots-clés associés, très pratiques pour avoir une idée de comment la reproduire : https://github.com/pharmapsychotic/clip-interrogator

### Mots-clés et artistes

Une liste de mots-clés qui fonctionnent bien en règle générale : https://moritz.pm/posts/parameters  
Une liste d’époques et de style artistique avec les noms d’artistes correspondants : https://docs.google.com/document/d/1SaQx1uJ9LBRS7c6OsZIaeanJGkUdsUBjk9X4dC59BaA/edit#heading=h.71g1pt84p9hx

### Conseils avancés

- L’ordre des mots (token) est important, toujours mettre le sujet en premier. Mettre des dates et époques peut aussi être important.
- Répéter des mots est utile pour forcer l'attention dessus
- Les parenthèses et crochets peuvent servir : (((mot))) augmente fortement l’attention portée et [mot] la diminue légèrement

## Construire d’une bonne image

Cinq Étapes pour construire une bonne image.

1. **Construire un prompt de base** : suivre la règle des 4 étapes (sujet, style, artiste, modificateurs), en ayant en amont cherché les bons mots-clés et l’inspiration sur Lexica.

**Etape 2 et 3 de façon cyclique: bon prompt -> bonne seed -> meilleur prompt -> meilleur seed ...**  

2. **Améliorer son prompt (A/B Testing)** : utiliser une seed fixée pour ajouter et supprimer des éléments de son prompt pour l’affiner.

3. **Trouver une bonne seed** : générer beaucoup d’images sur des seed différentes avec un sampler rapide (K_LMS, DDIM) en 8-20 steps jusqu’à trouver une seed satisfaisante.

4. **Raffiner l’image** : produire une image de qualité en modifiant légèrement son prompt et en utilisant le sampler k_lms en 100 steps ou k_dpm_2_a en 80-150 steps.

5. **Post-processing**: touche finale sur l’image pour corriger les derniers défauts (avec Photoshop par exemple ou en img2img). C’est aussi le moment de faire de l’upscale pour avoir une belle image finale !

# Conclusions

Fin du mini-guide, je le mettrais à jour si je trouve de nouvelles info importantes. Ce guide est bien entendu un gros plagiat de plein de sources que j'ai pu voir à droite à gauche et que j'ai sélectionner, c'est comme ça que fonctionne internet. A+
