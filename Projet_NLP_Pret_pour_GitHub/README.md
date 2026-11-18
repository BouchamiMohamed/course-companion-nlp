# Course Companion - Projet NLP 2

**Auteur : Mohamed Bouchami** - Master DSSD  
**Encadrant : Dr Riadh BOUSLIMI**

Assistant pédagogique local : import PDF/TXT, résumés de trois niveaux, QCM corrigés et chatbot RAG avec sources.

## Installation et lancement

Python 3.11 ou 3.12 et [Ollama](https://ollama.com/download) sont nécessaires.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
.venv\Scripts\python.exe -m streamlit run app.py
```

Ollama doit fonctionner à `http://localhost:11434`. Si nécessaire, lancer `ollama serve` dans un autre terminal.
Dans l'application, choisir **Ollama LLM** puis **Semantic embeddings (Ollama)** pour le pipeline hybride complet.
Le mode initial **Offline preview** affiche des extraits sans génération LLM. Le cours NLP de démonstration est inclus.
Importer un PDF lisible ou un texte UTF-8 ; les PDF scannés exigent un OCR préalable.

## Fonctionnalités

- Résumés Court / Moyen / Détaillé avec sections et sources lisibles.
- QCM Easy / Medium / Hard, quatre choix, correction, explication et export JSON.
- Recherche lexicale ou hybride, réponses ancrées dans des passages et leurs pages PDF.
- Onglet Library pour vérifier le texte réellement extrait.
- Messages de diagnostic et modes de secours explicitement signalés.

## Architecture et paramètres

`app.py` gère Streamlit. `study_assistant/corpus.py` extrait et segmente les cours ; `retrieval.py` construit
l'index vectoriel en mémoire ; `llm.py` appelle Ollama ; `services.py` orchestre la génération et la validation ;
`presentation.py` rend les références lisibles. Les modèles sont qwen2.5:3b et nomic-embed-text.
Les valeurs nominales sont 240 mots par passage, chevauchement 40, top-k 4 et seuil 0,08.
Le classement hybride combine 45 % de similarité dense et 55 % de similarité lexicale.
Les poids sont des choix du prototype, non des paramètres optimisés sur un benchmark indépendant.

## Vérifications

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe evaluate.py
.venv\Scripts\python.exe -m scripts.acceptance --semantic
```

Pour évaluer un autre cours : ajouter `--course "chemin\cours.pdf"` au script d'acceptation.
Les questions de chat préconfigurées pour le cas PDF correspondent au cours Oracle étudié ; adaptez-les à un autre sujet.
35 tests ont été confirmés le 18 novembre 2026. Les journaux historiques et les mesures recalculées sont dans `resultats/`.
Le journal d'acceptation contient un essai de retrieval antérieur à sa correction ; consultez aussi
`resultats/retrieval_verification.json` pour la vérification ciblée corrigée.

## Rapport

[Rapport détaillé en français](rapport/Rapport_Projet_NLP_Mohamed_Bouchami.pdf)

Le rapport décrit les mesures réelles, les exemples, les limites et le protocole d'évaluation humaine proposé.
L'évaluation indépendante par des étudiants ou l'enseignant n'a pas encore été collectée. Le dépôt reste à publier.
Après publication, vérifiez le lien dans le rapport et actualisez l'état de remise.

## Limites à connaître

Les tests valident des structures et des scénarios de panne ; ils ne prouvent pas une exactitude factuelle de 100 %.
La fidélité des résumés et des quiz doit être relue. Les longueurs restent approximatives.
Un secours extractif ou un exercice de rappel est signalé ; il n'est pas présenté comme une génération LLM normale.
Les imports sont conservés en mémoire de session, pas durablement. Chaque question du chat est traitée indépendamment.
Le modèle dense choisi est principalement orienté anglais ; la recherche en français nécessite une évaluation dédiée.
Le prototype n'inclut ni OCR, ni compréhension complète des figures/équations, ni authentification de production.
Le PDF de cours original, l'environnement `.venv` et les poids de modèles ne sont pas distribués.
Le dépôt contient des extraits de cours dans le rapport et les journaux ; vérifiez vos droits de diffusion avant publication publique.

## Informations complémentaires

- [Documentation technique détaillée en anglais](README_EN.md)
- [Guide de publication GitHub en français](PUBLICATION_GITHUB_FR.md)
- Profil de l'auteur : https://github.com/BouchamiMohamed
