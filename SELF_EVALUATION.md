# Self Evaluation

## What Works

- Persona drift is tracked per CSV row/day, not only as a global persona.
- Drift output includes timeline labels such as `Day 1 -> curious & formal` and a trigger object for topic/person/tone shift.
- Intent classification is fully offline using TF-IDF + LogisticRegression and is saved as a small pickle model.
- Conflict resolution ranks chunks by lexical relevance, recency, and emotional weight, then flags contradictory stances.
- Relationship questions such as `Did I mention anything about my sister?` can be answered through `/resolve-conflict` or `/chat`.

## Constraints Met

- No OpenAI/Gemini calls in the intent module.
- CPU-only intent inference.
- Model size is far below 50 MB for the current TF-IDF classifier.
- Retrieval remains local through TF-IDF + FAISS.

## Known Limitations

- Day detection assumes each CSV row is one chronological day because the provided schema has no explicit timestamp column.
- Drift trigger detection uses interpretable lexical/person heuristics; it is explainable but less nuanced than a supervised emotion model.
- Intent classifier is lightweight and fast, but accuracy depends on the weak labels and seed examples until a larger labeled dataset is added.
- The conflict resolver detects stance contradictions heuristically and should be expanded with more relation/event labels for production.

## Next Improvements

- Add timestamp parsing if the raw export includes dates.
- Add manually labeled validation data for all five intent classes.
- Store contradiction claims as structured triples: person, relation/event, sentiment, evidence message IDs.
- Add frontend panels for drift timeline, intent test box, and conflict evidence.
