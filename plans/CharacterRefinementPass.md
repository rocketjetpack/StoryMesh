# 📘 Pipeline Upgrade Spec: Narrative Depth & Completion

## Objective
Improve story outputs by enforcing:
- Character complexity
- Meaningful decisions
- Irreversible outcomes
- Concrete world grounding
- Narrative pressure

---

# 1. Character Contradiction Module

## Requirement
Each protagonist must contain **at least one active internal contradiction** that affects behavior.

### Definition
A contradiction = two traits/beliefs that would lead to **different decisions in the same situation**.

### Examples (valid)
- Values truth but prioritizes procedural correctness  
- Seeks justice but avoids conflict  
- Believes in systems but distrusts people inside them  

### Examples (invalid)
- “conflicted”  
- “unsure”  
- “has doubts”  

---

## Check
Prompt:
> List the protagonist’s top 3 traits. Identify at least one pair that would produce conflicting actions in a high-stakes situation.

### Fail Conditions
- No real tension between traits  
- Traits are redundant or aligned  

---

## Intervention
If failed:
- Inject a conflicting trait  
- Add or modify a scene where this contradiction **changes a decision or prevents action**

---

# 2. Decision With Cost Module

## Requirement
The story must include a **decision that incurs a non-reversible cost**.

### Valid Cost Types
- Loss of relationship  
- Career risk or consequence  
- Loss of access/information  
- Moral compromise  
- Foreclosed future option  

---

## Check
Prompt:
> What decision does the protagonist make that changes their situation, and what does it cost them?

### Fail Conditions
- Decision = observation, realization, or documentation only  
- Cost is vague or hypothetical  
- Outcome is reversible  

---

## Intervention
If failed:
- Add a decision point near the final third of the story  
- Attach a **specific consequence** that cannot be undone  
- Ensure the consequence persists into the ending  

---

# 3. Irreversibility Check (Ending Validator)

## Requirement
The ending must change the story state such that something is **no longer possible**.

---

## Check
Prompt:
> After the ending, what can the protagonist no longer do, access, or believe?

### Fail Conditions
- Answer is “nothing”  
- Only emotional tone has changed  
- Character could resume prior life unchanged  

---

## Intervention
If failed:
- Modify ending to include:
  - Loss of role, access, or relationship  
  - Irreversible knowledge that forces constraint  
  - Committed action with lasting consequence  

---

# 4. Concrete Mechanism Injection

## Requirement
Story must include at least one **specific operational detail** explaining how the system/world functions.

### Examples
- A bureaucratic loophole  
- A technical constraint  
- A procedural exploit  
- A financial or legal mechanism  

---

## Check
Prompt:
> Identify one concrete mechanism that enables the central conflict. How does it function step-by-step?

### Fail Conditions
- Explanation is vague or purely thematic  
- Mechanism is implied but not described  
- Could be replaced with “something happens”  

---

## Intervention
If failed:
- Add one explicit, grounded detail showing how the system operates  
- Ensure it appears **in-scene**, not just exposition  

---

# 5. Asymmetry Injection Module

## Requirement
Introduce at least one **intentional misalignment** in the story.

### Types of Asymmetry
- A character who interprets events differently  
- A tonal shift (e.g., moment of bluntness in otherwise quiet prose)  
- A scene that disrupts pacing or expectation  
- A partial or misleading explanation  

---

## Check
Prompt:
> Where does the story introduce tension through inconsistency or competing interpretations?

### Fail Conditions
- All elements reinforce the same interpretation  
- No competing perspectives or tonal variation  

---

## Intervention
If failed:
- Add a secondary character or moment that challenges the dominant framing  
- Introduce a scene with a different emotional or narrative texture  

---

# 6. State Change Tracking

## Requirement
The protagonist must undergo a **clear state transition**.

---

## Check
Prompt:
> Define:
> - Initial state (beliefs, constraints, options)  
> - Final state (beliefs, constraints, options)

### Fail Conditions
- Initial and final states are functionally identical  
- Only superficial emotional change  

---

## Intervention
If failed:
- Modify ending or key decision so that:
  - Options are reduced or altered  
  - Beliefs are updated in a way that affects future behavior  

---

# 7. Pressure Gradient Module

## Requirement
The story must demonstrate **increasing constraint or stakes** over time.

---

## Check
Prompt:
> For each major section, describe how pressure increases (risk, constraint, clarity, or consequence).

### Fail Conditions
- Flat tension profile  
- Repetition without escalation  
- Stakes introduced but not intensified  

---

## Intervention
If failed:
- Add escalation via:
  - New information that raises stakes  
  - Reduced options for the protagonist  
  - External response from institutions/characters  

---

# 8. Narrative vs Aesthetic Completion Check

## Requirement
The story must include a **material change**, not just thematic closure.

---

## Check
Prompt:
> What happens in the final 10% of the story that materially changes the situation?

### Fail Conditions
- Ending consists only of reflection or realization  
- No new action or consequence  
- Situation remains static  

---

## Intervention
If failed:
- Introduce a final action, decision, or event  
- Ensure it affects:
  - Character options  
  - Relationships  
  - Knowledge or access  

---

# 9. Failure Mode Detection

## Monitor For:

### Failure Mode A: Elegant Stasis
- High thematic coherence  
- No real decision or consequence  

### Failure Mode B: Interpretive Overload
- Excessive implication  
- Insufficient concrete detail  

---

## Intervention
- For A → enforce Decision With Cost  
- For B → enforce Concrete Mechanism Injection  

---

# 10. Global Validation Rule

## Every story must include:

1. **A contradiction in the protagonist**  
2. **A decision that incurs cost**  
3. **A change that cannot be undone**  

---

## Final Check
Prompt:
> Verify that all three conditions are satisfied. If not, revise the story.

### Fail Condition
- Any one of the three is missing  

---

## Outcome
A story that is:
- Not just coherent and well-written  
- But **pressurized, consequential, and complete**