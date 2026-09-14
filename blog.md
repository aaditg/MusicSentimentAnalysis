# Reading the mood of a song from three angles

I spent the last few weeks building a system that guesses the emotion of a song. Not the genre, not the artist — the *feeling*. Happy and loud, sad and slow, angry, calm. It turns out you can get surprisingly far with this, and the interesting part isn't the final accuracy number. It's what each part of a song gives away, and what it hides.

## The four-quadrant idea

Instead of trying to label a song "happy" or "melancholic" directly, I used the circumplex model from psychology: two axes, valence (how positive the emotion is) and arousal (how energetic it is). Cross them and you get four quadrants:

- Q1 — high valence, high arousal: upbeat, joyful, energetic
- Q2 — low valence, high arousal: angry, tense, aggressive
- Q3 — low valence, low arousal: sad, tired, gloomy
- Q4 — high valence, low arousal: calm, tender, content

Every model in the project predicts one of these four labels. It keeps things comparable across very different inputs.

## Three ways to look at a song

A song isn't one thing. It's a score, a recording, and a set of words, and each of those is a different dataset with a different personality.

**Symbolic (the notes themselves).** Using EMOPIA, a set of piano performances in MIDI, I pulled out things like pitch range, note density, velocity, key and mode, and a pile of rhythm and interval statistics. A logistic regression on these lands around 0.64 macro-F1. Respectable, but it plateaus hard — more on that below.

**Lyrics (the words).** Using the MERGE lyrics set, plain TF-IDF features over words and character n-grams feed a linear SVM. After tuning, that reaches 0.69 macro-F1. Words carry a lot of valence: the model keys on exactly the terms you'd expect, aggression and profanity for Q2, "alone", "tired", "cold" for the sad quadrant.

**Audio (the recording).** Using DEAM, I extracted acoustic features and this is where it gets hard. Audio is great at arousal — loud and fast is easy to hear — but poor at valence. A plain classifier just collapses on the two "disagreement" quadrants, scoring near-zero F1 on them. The best I managed was around 0.49 macro-F1, and it took real work to get there.

## Fixing the audio collapse

The failure mode was worth understanding rather than papering over. Audio hears energy but not mood, so Q2 (angry) and Q4 (calm) — the quadrants where valence and arousal disagree — are where everything falls apart.

Two things helped. First, instead of classifying the quadrant directly, I trained two regressors to predict valence and arousal, then read the quadrant off the predictions. Second, I used SMOTE to synthetically rebalance the minority quadrants. Both push macro-F1 up to roughly 0.49 and, more importantly, stop the model from ignoring the hard classes entirely. It's still the weakest modality. That's honest: you cannot reliably hear happy-versus-sad from audio features alone.

![Best macro-F1 by modality](results/figures/best_per_modality.svg)

## Two modalities beat one

The MERGE bimodal set pairs audio and lyrics for the same songs, so I could test the obvious question: does combining them help? Yes, clearly. Lyrics alone score 0.68, audio alone 0.63, and fusing them jumps to 0.74. A weighted late fusion — train each modality separately and blend their scores — did best at 0.743, leaning about 60% on lyrics. The two modalities are complementary: audio rescues some of the calm/energetic distinction that lyrics miss.

## Then the language models showed up

The classical lyrics model tops out around 0.69. Fine-tuning DistilBERT on the same split pushed that to 0.72. But the real jump came from just asking a large model. GPT-5.5, given nothing but the four definitions and the lyrics, zero-shot, hit 0.76 — the best lyrics-only result in the whole project, with no training at all. Even the small gpt-4.1-mini matched the fine-tuned transformer out of the box.

One counterintuitive result: adding a few labeled examples to the prompt (few-shot) made both models slightly *worse*. Their prior knowledge of emotion in language is already strong enough that four hand-picked examples just added noise.

## The things that didn't work

The negative results taught me the most.

I nearly doubled the symbolic feature set — pitch-class histograms, melodic direction, articulation, polyphony, the works — expecting a jump. It moved the needle by almost nothing. The song-level key and mode already carried most of the separable signal; the fancy note statistics were mostly redundant. I did find and fix a genuine bug in the process (the tempo was hardcoded, so every duration feature was silently computed at the wrong speed), but the accuracy stayed flat. Sometimes more features is just more features.

I also tried learning the valence/arousal decision thresholds per fold instead of splitting at the midpoint. It slightly hurt. The midpoint was already about right.

## One label per song is the wrong shape

Everything above assigns a song a single quadrant. That works for the datasets, because the
annotations describe 30-second excerpts — but it's a strange thing to do to a song that goes
somewhere. Pink Floyd's "Shine On You Crazy Diamond" spends four minutes on an ambient synth wash
before the first guitar phrase and doesn't reach a vocal until past the eight-minute mark. Asking
for *the* emotion of that is asking the wrong question.

The fix falls out of the dataset's own design. Since the model was fitted on 30-second excerpts, a
30-second window is exactly the granularity it understands. Slide that window across a full track
and every prediction stays in-distribution — you get a trajectory instead of a label, without
changing the model at all.

To plot it, I project the four quadrant scores back onto the axes they were built from:

    valence = (Q1 + Q4) - (Q2 + Q3)
    arousal = (Q1 + Q2) - (Q3 + Q4)

Those two curves over time are the song's emotional arc: a quiet intro sits low on both, a chorus
pushes arousal up, a turn to a minor key drags valence down.

One honest caveat. Lyrics aren't time-aligned to audio here, so the lyrics branch sees the whole
text no matter which window it scores — it shifts every point by the same constant. The *shape* of
the trajectory always comes from audio. Which is a little ironic, since audio is the weakest
modality on static classification. It turns out to be the only one that can tell you *when*.

## Where it lands

Across everything, one weak spot is universal: Q4, the calm-and-content quadrant, is the hardest for every model, classical or neural or LLM. It's the quietest emotion and the easiest to confuse. And the ranking of lyrics models on a fair, held-out split tells the whole arc of machine learning in one line:

TF-IDF SVM 0.665 → fine-tuned DistilBERT 0.716 → GPT-5.5 zero-shot 0.764.

![Three eras of lyrics classification](results/figures/lyrics_arc.svg)

Decades of feature engineering, then fine-tuning, then a model that just knows. The full numbers are in `results/` if you want to dig in.
