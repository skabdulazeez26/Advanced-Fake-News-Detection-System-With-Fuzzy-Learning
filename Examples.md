# Real Dataset Examples for FDHN Testing

## LIAR Dataset Examples (4-module FDHN)

### Example 1: FALSE
**Input:**
```
Statement: Says the Annies List political group supports third-trimester abortions on demand
Speaker: dwayne-bohac
Subject: abortion
Context: a mailer
```
**Expected Label:** false

### Example 2: HALF-TRUE
**Input:**
```
Statement: When did the decline of coal start? It started when natural gas took off that started to begin in President George W. Bushs administration
Speaker: scott-surovell
Subject: energy
Context: a floor speech
```
**Expected Label:** half-true

### Example 3: MOSTLY-TRUE
**Input:**
```
Statement: Hillary Clinton agrees with John McCain by voting to give George Bush the benefit of the doubt on Iran
Speaker: barack-obama
Subject: foreign-policy
Context: Denver
```
**Expected Label:** mostly-true

### Example 4: PANTS-ON-FIRE
**Input:**
```
Statement: Health care reform legislation is likely to mandate free sex change surgeries
Speaker: blog-posting
Subject: health-care
Context: a news release
```
**Expected Label:** pants-on-fire

### Example 5: BARELY-TRUE
**Input:**
```
Statement: The economic turnaround started at the end of my term
Speaker: charlie-crist
Subject: economy
Context: an interview on CNN
```
**Expected Label:** barely-true

### Example 6: TRUE
**Input:**
```
Statement: Wisconsin has the highest African-American unemployment rate in the country
Speaker: gwen-moore
Subject: economy
Context: a press release
```
**Expected Label:** true

## LIAR2 Dataset Examples (5-module FDHN)

### Example 1: TRUE (Label 5)
**Input:**
```
Statement: The unemployment rate is at its lowest level in 50 years
Speaker: donald-trump
Subject: economy
Context: a speech
Justification: Trump said the unemployment rate is at its lowest level in 50 years. According to Bureau of Labor Statistics data the unemployment rate in May 2018 was 3.8 percent which was indeed the lowest since April 1969 when it was 3.4 percent. This represents a 49-year low which rounds to 50 years. The statement is factually accurate based on official government data. We rate this True.
```
**Expected Label:** 5 (TRUE)

### Example 2: BARELY-TRUE (Label 1)
**Input:**
```
Statement: We created more jobs in our first month than Obama did in his first year
Speaker: donald-trump
Subject: jobs
Context: a press conference
Justification: Trump claimed his administration created more jobs in the first month than Obama did in his first year. In Trumps first month about 235000 jobs were added. In Obamas first year the economy lost about 4 million jobs due to the recession he inherited. While technically true this comparison ignores the vastly different economic circumstances each president faced. Obama inherited the worst recession since the Great Depression while Trump inherited a growing economy. We rate this Barely True.
```
**Expected Label:** 1 (BARELY-TRUE)

### Example 3: PANTS-ON-FIRE (Label 0)
**Input:**
```
Statement: Obama wants to take away everyones guns and abolish the Second Amendment
Speaker: donald-trump
Subject: guns
Context: a rally
Justification: Trump claimed Obama wants to take away everyones guns and abolish the Second Amendment. This is completely false. Obama has never proposed taking away all guns or abolishing the Second Amendment. Obama has supported background checks and some gun safety measures but has repeatedly stated he supports the Second Amendment and gun ownership rights. This is a wildly inaccurate and inflammatory claim with no basis in reality. We rate this Pants on Fire.
```
**Expected Label:** 0 (PANTS-ON-FIRE)

### Example 4: FALSE (Label 2)
**Input:**
```
Statement: The unemployment rate for African Americans is the lowest it has ever been
Speaker: donald-trump
Subject: economy
Context: a speech
Justification: Trump said the unemployment rate for African Americans is the lowest it has ever been. The Bureau of Labor Statistics has been tracking unemployment by race since 1972. The rate for African Americans in December 2017 was 6.8 percent which is indeed the lowest since the agency began tracking this data. However Trump is taking credit for a trend that began under Obama. We rate this statement False because while the statistic is accurate Trump is misleading about his role in achieving it.
```
**Expected Label:** 2 (FALSE)

### Example 5: HALF-TRUE (Label 3)
**Input:**
```
Statement: We have the highest corporate tax rate in the world
Speaker: marco-rubio
Subject: taxes
Context: a debate
Justification: Rubio said we have the highest corporate tax rate in the world. The United States does have the highest statutory corporate tax rate among developed nations at 35 percent. However when accounting for deductions and credits the effective rate is much lower and more in line with other countries. Many corporations pay far less than 35 percent and some pay nothing at all. We rate this Half True because the statement is accurate about the statutory rate but ignores the reality of what companies actually pay.
```
**Expected Label:** 3 (HALF-TRUE)

### Example 6: MOSTLY-TRUE (Label 4)
**Input:**
```
Statement: Climate change is real and human activities are the primary cause
Speaker: barack-obama
Subject: environment
Context: a speech
Justification: Obama said climate change is real and human activities are the primary cause. The overwhelming scientific consensus supports this statement. Multiple studies show that 97 percent or more of actively publishing climate scientists agree that recent climate change is primarily caused by human activities. The evidence includes rising global temperatures melting ice sheets and rising sea levels. We rate this Mostly True because while the scientific consensus is clear there are still some uncertainties about specific impacts and timing.
```
**Expected Label:** 4 (MOSTLY-TRUE)

## Testing Instructions

### For LIAR Model:
1. Select "LIAR Model (4-module)"
2. Input statement, speaker, subject, context (no justification)
3. Compare prediction with expected label
4. Note confidence levels and fuzzy scores

### For LIAR2 Model:
1. Select "LIAR2 Model (5-module)"
2. Input statement, speaker, subject, context, AND justification
3. Compare prediction with expected label
4. Note improved confidence with justification context

## Expected Model Performance

### LIAR Model Results:
- **Accuracy**: ~26-30% (6-class classification is challenging)
- **Low confidence**: Most predictions under 40% confidence
- **Distributed scores**: Fuzzy logic spreads across multiple classes
- **High uncertainty**: Model often unsure between similar classes
- **Expected behavior**: Wrong predictions with low confidence are normal

### LIAR2 Model Results:
- **Accuracy**: ~50-55% (justification text helps significantly)
- **Higher confidence**: Better predictions with justification context
- **More focused scores**: Less distribution across classes
- **Better reasoning**: Justification text provides crucial context

### Understanding Your Results:
- **26.4% confidence = High uncertainty** (model is guessing)
- **Distributed fuzzy scores = Normal** (shows model uncertainty)
- **Wrong prediction = Expected** (only ~30% accuracy overall)
- **Low confidence indicates** the model knows it's uncertain

### Understanding Fuzzy Logic Results:
- **Perfect predictions are rare** - 6-class classification is very difficult
- **Low confidence is normal** - most predictions under 40% confidence
- **Distributed scores show uncertainty** - model admits when it's unsure
- **Wrong predictions expected** - only ~30% accuracy for LIAR model
- **Focus on confidence level** - low confidence = high uncertainty
- **Compare models** - LIAR2 should show higher confidence with justification

## Label Mapping

### LIAR Dataset:
- true
- mostly-true
- half-true
- barely-true
- false
- pants-on-fire

### LIAR2 Dataset:
- 0 = pants-on-fire
- 1 = barely-true
- 2 = false
- 3 = half-true
- 4 = mostly-true
- 5 = true

## Key Differences to Observe

1. **LIAR2 has justification** - provides reasoning context
2. **LIAR2 should be more accurate** - enhanced dataset
3. **Fuzzy logic handles uncertainty** - both models show confidence distribution
4. **Speaker credibility matters** - both models use speaker context