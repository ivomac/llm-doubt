# Prompt-Induced Answer Abstention in LLMs on Obscure Factual Questions

## Abstract

I tested 9 LLMs on 247 obscure Wikipedia-derived factual questions under two system prompts: a baseline that specifies an answer format, and one that additionally instructs the LLM to leave the answer blank when they do not know the answer. A judge model classified responses as CORRECT, INCORRECT, or DOUBT. Under the second prompt, incorrect answers transitioned to DOUBT at a higher rate than correct answers (aggregate difference: 0.079, 95% CI [0.039, 0.118]), with substantial variation across models. We discuss several possible interpretations of this shift, and outline what additional experiments would be needed to distinguish them.

## Motivation

Through training on internet data repeatedly, LLMs have memorized many unusual facts. However, they are compression machines and will lose factual accuracy, especially on more fringe facts, such as what we can find when jumping to [random Wikipedia pages](https://en.wikipedia.org/wiki/Special:Random).

In the space of fringe facts, I found anecdotally that a minimal prompt change can produce drastically different behavior. Below, Claude Sonnet 4 first correctly answers the 10 millionth prime (179,424,673), but with a simple addition suggesting the answer may be blank, it denies knowing the number, despite having just demonstrated knowing it exactly:

[https://claude.ai/share/29db97e4-517f-49a9-a5a2-5dd6fcbd2d83](https://claude.ai/share/29db97e4-517f-49a9-a5a2-5dd6fcbd2d83)
![][claude_knows]

[https://claude.ai/share/a53f6df3-7249-4f08-8bdb-6f5593412a4e](https://claude.ai/share/a53f6df3-7249-4f08-8bdb-6f5593412a4e)
![][claude_doubts]

## Setup

### Dataset

At first I planned to focus on mathematical facts like above, but it seemed hard to automate the construction of a good dataset of such questions, so instead I turned to very specific facts one can find on Wikipedia, such as:

* "Where is the Syrian Writers Association registered? [United Kingdom](https://en.wikipedia.org/wiki/Syrian_Writers_Association)"
* "In what century did European art begin depicting Jesus with visible genitalia? [13th century](https://en.wikipedia.org/wiki/Sexuality_and_marital_status_of_Jesus)"

Using Anthropic's API ("Claude Haiku 3.5") and a simple script, I built a dataset of similar questions. Haiku 3.5 is highly accurate at extracting specific facts from provided context, so the generated question-answer pairs are reliable even though produced by an LLM. The answers are short and mostly years, dates, places, people, buildings…

### System Prompt

Through some experimentation I saw that the addition of "or blank" in the prompt as shown above did not produce a consistent effect across models, so I used more explicit system prompts for my final run:

#### "BASE" SYSTEM PROMPT

```
When asked a question, write your final answer at the end of your message in this format:

Answer: your answer here
```

#### "SUGGEST EMPTY" SYSTEM PROMPT

```
When asked a question, write your final answer at the end of your message in this format:

Answer: your answer here

If you do not know the answer, leave the final answer empty, like this:

Answer:
```

### Tested LLMs

I queried 9 dense open-source models directly available on [together.ai](http://together.ai):

* gpt-oss-20b
* gpt-oss-120b

* Meta-Llama-3.1-8B-Instruct-Turbo
* Meta-Llama-3.1-70B-Instruct-Turbo
* Meta-Llama-3.1-405B-Instruct-Turbo

* Qwen2.5-7B-Instruct-Turbo
* Qwen2.5-72B-Instruct-Turbo

* DeepSeek-R1-Distill-Qwen-14B
* DeepSeek-R1-Distill-Llama-70B

For consistency I used a temperature of 0.01.

The total API costs, including experimentation and classification, were around $15.

### Classification

I used OpenAI's "gpt-oss-120b" model as a judge to classify the answers as "CORRECT", "INCORRECT", and "DOUBT".

Initially the category "DOUBT" was named "BLANK", but several queried models did not follow what is specified in the system prompt to not answer, instead giving answers such as "Answer: unknown" or "Answer: It could be around 400, but I would need to search to know exactly.". "DOUBT" better represents this category of answers and helped the judge in the classification.

Smaller models would sometimes get stuck repeating the same sentences or analysis, producing responses classified as ERROR. These specific combinations were re-run manually with increasing temperature (0.01 → 0.5) until a valid category was reached. Each combination was only run once per temperature setting, no multiple repetitions or voting were used.

The judge receives only the extracted answer (the final "Answer:" line), not the full model response.

### Bootstrap for confidence interval

To determine whether the observed transition rate differences are statistically significant, I use a bootstrap resampling procedure:

1. Resample 247 questions with replacement (preserving all 9 model observations per question to maintain within-question correlation)
2. Compute the row-normalized transition matrix on the resampled data
3. Repeat 8,000 times
4. Report the 2.5th and 97.5th percentiles as the 95% confidence interval

## Results

We first look at the effect of the "suggest empty" prompt on the fraction of answers of each category, per model. In the plots below, we plot a line between the fractions for each prompt setup. A green/red line means the value increased/decreased for "suggest empty".

Overall, the "suggest empty" prompt affected the Llama3.1-70B and Qwen2.5-72B models the most, with both fractions of correct and incorrect answers decreasing significantly. The Llama and Qwen series of models already had a tendency to "DOUBT" above the other models, and were the most sensitive to the change in prompt.

On the other hand, OpenAI's models had a very small increase in "DOUBT" answers with "suggest empty", with gpt-oss-120b almost always giving an answer, regardless of prompt.

Deepseek's models seem closer to OpenAI's models than the other two series.

![][fractions]

From the results above alone we can't draw the conclusion right away that the LLMs are more likely to "DOUBT" when they don't know the answer to the question, since the fractions of "INCORRECT" are so high to begin with. For that, I show per-model counts and relative changes from "base" to "suggest":

| Model | # Params | INCORRECT (base→suggest) | DOUBT (base→suggest) | CORRECT (base→suggest) | Δ Accuracy |
| :---- | :---- | :---- | :---- | :---- | -----: |
| deepseek | 14B | 223→219 (−1.8%) | 4→7 (+75.0%) | 20→21 (+5.0%) | +6.3% |
|  | 70B | 171→153 (−10.5%) | 10→29 (+190.0%) | 66→65 (−1.5%) | +7.1% |
| gpt | 20B | 185→184 (−0.5%) | 36→38 (+5.6%) | 26→25 (−3.8%) | −2.9% |
|  | 120B | 193→182 (−5.7%) | 0→7 (+inf%) | 54→58 (+7.4%) | +10.5% |
| llama | 8B | 153→111 (−27.5%) | 60→104 (+73.3%) | 34→32 (−5.9%) | +23.1% |
|  | 70B | 95→67 (−29.5%) | 82→121 (+47.6%) | 70→59 (−15.7%) | +10.4% |
|  | 405B | 43→37 (−14.0%) | 134→144 (+7.5%) | 70→66 (−5.7%) | +3.4% |
| qwen | 7B | 210→183 (−12.9%) | 22→48 (+118.2%) | 15→16 (+6.7%) | +20.6% |
|  | 72B | 147→85 (−42.2%) | 65→136 (+109.2%) | 35→26 (−25.7%) | +21.8% |

Apart from gpt-20B, all other models had a bigger relative decrease in "INCORRECT" questions than "CORRECT".

Below I show how the answers across all models changed categories when going from "base" to "suggest empty". Unsurprisingly, the largest change was from "INCORRECT" to "DOUBT",  where the biggest contributor was Qwen2.5-72B.

|  | → CORRECT | → DOUBT | → INCORRECT | Total |
| :---- | ---: | ---: | ---: | ---: |
| CORRECT | 308 | 37 | 45 | 390 |
| DOUBT | 8 | 349 | 56 | 413 |
| INCORRECT | 52 | 248 | 1120 | 1420 |

A bootstrap confidence interval analysis (10,000 resamples at the question level) confirms the distinction is statistically significant:

- **INC → D**: 0.174 [0.152, 0.197] - answers that were incorrect under the base prompt are doubted at ~17% rate under "suggest empty"
- **C → D**: 0.095 [0.064, 0.131] - answers that were correct under the base prompt are doubted at only ~10% rate
- **Difference**: 0.079 [0.039, 0.117] - 95% CI entirely above zero, confirming INC is more likely to flip to DOUBT than CORRECT

The per-model plot below shows the difference for each model alongside the aggregate (red diamond). Per-model bootstrap CIs (1,000 resamples) reveal considerable variation.

The two gpt-oss models show no evidence of an effect (CIs include zero). The largest differences are in llama-405B and qwen-72B, which also have the highest baseline accuracy. The aggregate result is driven primarily by a subset of sensitive models, not a uniform property of all LLMs.

![][per_model_ci]

### Questions analysis

Taking a specific question, I count how many times each base to suggest transition occurred across the 9 models. Sorting by this count, the top 19 counts are (INCORRECT → DOUBT). Here are the top 10 counts:

* 5 times:
  * What is the specific point of land named after the first European to explore the Hanover area?
* 4 times:
  * From which university did Guo Jiakun graduate in 2002?
  * On what date was the men's K-1 500 metres sprint canoeing competition held at the 2002 Asian Games?
  * On what date did the 6th Army first encounter Soviet defensive lines during Operation Fischreiher?
  * In what year did Ibrahim I ibn al-Aghlab name Abu Muhriz as Qadi of Kairouan?
* 3 times:
  * How much money did the UK government spend on focus groups and polling to promote the Eat Out to Help Out scheme?
  * In which shrine did Nishikifuji Ryūsei get married in Tokyo?
  * In which specific year did James Phelan kick a 61-yard field goal attempt that bounced off the crossbar?
  * In what season of its history was Aswan SC during the 2020-21 season?
  * Which character is responsible for the 1883 eruption of Krakatoa in "The Dresden Files"?

To be clear, this means that 5 models (out of 9) got the first question wrong in "base" and changed to "DOUBT" in "suggest empty". I don't notice any particular pattern among these questions that distinguishes them from the rest.

## Discussion

The main result is that incorrect answers are more likely than correct ones to flip to DOUBT under the suggest_empty prompt, at least for random wikipedia facts. What is less clear is why.

A simple reading of the base system prompt is that it influences formatting only. The base prompt says "Answer: your answer here", indicating a template, commonly used for ease of parsing. The suggest_empty prompt extends the template with a conditional: if you don't know, leave it blank. The simplest explanation of the observations, of which I was biased from the beginning to push for, is that the main effect comes not from the formatting, but from the presence of the expression "if you do not know", which more often than not triggers the model to consider whether it actually knows.

However, the base prompt is also pushing models to give a definite answer, beyond their inclination when an answer format is not imposed, which more often than not implies that a concrete answer is preferred by the user than no answer at all. In contrast, the suggest_empty prompt makes it clear that the user does not mind an uncommitted answer, removing pressure from the model. I this is the case, it would still be an interesting find, and a warning of how subtle prompt changes can influence the model's completions by indicating different user preferences.

This ambiguity could only be solved by testing additional variations of each prompt, using different phrasings to specify format, or, ideally, skipping format imposition entirely. The lack of prompt averaging is a limitation of this study. Another limitation is the llm-as-a-judge, whose classification accuracy was not thoroughly evaluated: the judge model used is one of the studies models, introducing bias

Note that INCORRECT→DOUBT exceeds CORRECT→DOUBT rate-wise, suggesting that some models can, on average, distinguish between their own wrong/correct answers. One possibility is that models possess some internal signal that correlates with answer correctness, and the suggest_empty prompt gives them a way to express it. The model's internal uncertainty is higher for questions it gets wrong, and given the option to abstain, it does so selectively. To complement this analysis, a set of less obscure facts should have been tested on as well, where we would expect to see no significant effect between the two studied prompts.

## Future Work

In this mini-project we only classified the final answer, but the initial 10 millionth prime example shows drastically different justifications for the different answers. Extending the analysis to the reasoning traces that result in different answers would be more informative. The original idea of testing math questions remains to be explored. It could give different results compared to Wikipedia fact questions.

This study would be much stronger if we had looked instead at the next-token probability distribution at the "Answer:" stage. This would give higher granularity to the analysis and could help distinguish between the interpretation accounts discussed above.

### Gamification

I had a few other ideas to explore how to manipulate answer behavior, in particular by:

- Penalizing mistakes, while allowing abstention as an "easy way out" with a known point value.
- Providing fake statistics that the model is performing well or badly, to test whether external performance signals influence abstention decisions.

#### Simple Scoring

Prompt:

```
"Answer this question:

{question/math prob here}

You get the following points depending on your answer:

Correct: +1
Incorrect: {N (<1)}
```

We did not state explicitly what the "goal" is. Two possible views are:

* Try to answer the question correctly, disregarding the prompt above.
* Try to maximize the score.

In this case they are equivalent, but the model's answer could show sensitivity to N.
Does it abstain more as N decreases?

#### Say you don't know

We can extend the test to distinguish the two goals by scoring blank answers separately:

```
No Answer: {B (N <= B < 1)}
```

We have seen that suggesting an empty answer can have a measurable effect, but:

* Does the model try to optimize for score somehow?
* How does the answer depend on B and N?
* Do we see behavior transitions at specific values of B and N?

#### Performance/Success Metric

We could further add a "current total score" to the prompt. Is the model riskier if the total score is much larger than what is lost on a wrong answer, or vice versa?

A complete prompt would look like:

```
Question: {question}

You get the following points depending on your answer:

Correct: +1
Incorrect: N (N < 1)
No Answer: B (N <= B < 1)

Your current total score: T   OR   Your current success rate: P%

Write your answer at the end of your message in this format:

Answer: {answer or blank}
```

Of course, these are too many variables to test at the same time, going one by one would be the way to go.

[claude_knows]: images/claude_knows_10m_prime.png
[claude_doubts]: images/claude_doubts_10m_prime.png
[fractions]: images/fractions.png
[per_model_ci]: images/per_model_ci.png
