import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import google.generativeai as genai
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

GOOGLE_API_KEY = "your_api_key_here"  # ← PUT YOUR KEY HERE
genai.configure(api_key=GOOGLE_API_KEY)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'bert-base-uncased'

# Confidence threshold - trust BERT above this
BERT_CONFIDENCE_THRESHOLD = 0.75  # 75%

# ============================================================
# BERT MODEL ARCHITECTURE
# ============================================================

class AnswerEvaluationModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_features=3, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.attention = nn.Linear(self.hidden_size, 1)
        self.shared = nn.Sequential(
            nn.Linear(self.hidden_size + 32, 256), nn.LayerNorm(256),
            nn.ReLU(), nn.Dropout(dropout)
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
    
    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs.last_hidden_state
        att_weights = torch.softmax(self.attention(seq), dim=1)
        attended = torch.sum(att_weights * seq, dim=1)
        feat_out = self.feature_net(features)
        combined = torch.cat([attended, feat_out], dim=1)
        shared = self.shared(combined)
        score = self.regressor(shared).squeeze(-1)
        label = self.classifier(shared)
        return score, label

# ============================================================
# LOAD MODEL
# ============================================================

print("🔄 Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
checkpoint = torch.load('best_model_final.pt', weights_only=False, map_location=device)
model = AnswerEvaluationModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

def extract_keywords(text):
    """Extract meaningful keywords from text"""
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'to', 'and',
        'in', 'on', 'at', 'by', 'for', 'with', 'from', 'as', 'or', 'but',
        'if', 'then', 'so', 'than', 'that', 'this', 'these', 'those', 'it',
        'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom',
        'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'just', 'also', 'very', 'you', 'your', 'use',
        'using', 'used', 'because', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'further', 'once', 'here', 'there', 'any', 'many', 'much', 'one',
        'two', 'three', 'first', 'second', 'third', 'last', 'next'
    }
    
    words = text.lower().split()
    keywords = [w.strip('.,!?;:()[]"\'') for w in words 
                if w.strip('.,!?;:()[]"\'') not in stopwords 
                and len(w.strip('.,!?;:()[]"\'')) > 3]
    return set(keywords)

def concept_coverage(student, reference):
    """Calculate keyword coverage"""
    ref_keywords = extract_keywords(reference)
    student_keywords = extract_keywords(student)
    
    if len(ref_keywords) == 0:
        return 0
    
    matches = len(ref_keywords & student_keywords)
    return matches / len(ref_keywords)

def semantic_similarity(student, reference):
    """Calculate Jaccard similarity"""
    ref_keywords = extract_keywords(reference)
    student_keywords = extract_keywords(student)
    
    if not ref_keywords or not student_keywords:
        return 0
    
    intersection = len(ref_keywords & student_keywords)
    union = len(ref_keywords | student_keywords)
    
    return intersection / union if union > 0 else 0

def detect_wrong_concepts(student, reference):
    """Detect unrelated/wrong concepts"""
    ref_keywords = extract_keywords(reference)
    student_keywords = extract_keywords(student)
    
    extra_words = student_keywords - ref_keywords
    
    wrong_indicators = {
        'python', 'library', 'libraries', 'coding', 'code', 'answer',
        'animals', 'breathing', 'lungs', 'digestive', 'spam', 'email',
        'basically', 'stuff', 'things', 'whatever', 'something', 'somehow'
    }
    
    wrong_found = extra_words & wrong_indicators
    return len(wrong_found), list(wrong_found)[:3]

# ============================================================
# GROK FEEDBACK GENERATOR
# ============================================================

def generate_grok_feedback(question, reference_answer, student_answer, 
                           score, label, present, missing, kw_cov, sim):
    """Generate personalized AI feedback"""
    
    try:
        model_names = ['gemini-2.5-flash-lite']
        
        label_names = ['Incorrect', 'Partial', 'Correct']
        
        prompt = f"""You are an experienced, encouraging teacher providing feedback on a student's exam answer.

**Question:** {question}
**Reference Answer:** {reference_answer}
**Student's Answer:** {student_answer}

**Assessment:**
- Score: {score:.1f}/5.0
- Classification: {label_names[label]}
- Keyword Coverage: {kw_cov*100:.1f}%
- Semantic Similarity: {sim*100:.1f}%
- Concepts Present: {', '.join(list(present)[:6]) if present else 'None'}
- Missing Concepts: {', '.join(list(missing)[:6]) if missing else 'All covered'}

Write 3-4 sentences of constructive feedback:
1. Start with something positive
2. Explain what's good or missing (use specific concepts)
3. Give ONE actionable improvement tip
4. Be encouraging and professional

Feedback:"""

        for model_name in model_names:
            try:
                model_gemini = genai.GenerativeModel(model_name)
                response = model_gemini.generate_content(prompt)
                return response.text.strip()
            except:
                continue
        
        return "[AI feedback temporarily unavailable]"
        
    except Exception as e:
        return f"[AI feedback error: {str(e)[:50]}...]"

# ============================================================
# SMART GRADING FUNCTION
# ============================================================

def grade_answer(question, reference_answer, student_answer):
    """
    Smart grading that balances BERT confidence with rule-based checks
    
    Logic:
    1. Calculate features (keywords, similarity)
    2. Get BERT model prediction + confidence
    3. Decision:
       - BERT very confident (>75%) → TRUST BERT (it understands semantics)
       - BERT uncertain (<75%) → USE RULES to help decide
       - BOTH metrics < 30% AND BERT not confident about Correct → INCORRECT
    """
    
    if not question or not reference_answer or not student_answer:
        return ("❌ **Please fill all fields!**", "", "", "", "", "")
    
    # ============================================================
    # STEP 1: CALCULATE FEATURES
    # ============================================================
    
    kw_coverage = concept_coverage(student_answer, reference_answer)
    sim = semantic_similarity(student_answer, reference_answer)
    wrong_count, wrong_words = detect_wrong_concepts(student_answer, reference_answer)
    
    student_len = len(student_answer.split())
    ref_len = len(reference_answer.split())
    length_ratio = student_len / ref_len if ref_len > 0 else 0
    
    # ============================================================
    # STEP 2: GET BERT MODEL PREDICTION
    # ============================================================
    
    model_lr = min(length_ratio, 1.0)
    
    text = f"Question: {question} | Reference: {reference_answer} | Student: {student_answer}"
    enc = tokenizer(text, max_length=384, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        score_pred, label_logits = model(
            enc['input_ids'].to(device),
            enc['attention_mask'].to(device),
            torch.tensor([[kw_coverage, sim, model_lr]], dtype=torch.float).to(device)
        )
        
        model_score = score_pred.item() * 5
        model_label = torch.argmax(label_logits, dim=1).item()
        probs = torch.softmax(label_logits, dim=1)[0].cpu().numpy()
    
    # Get max confidence and which label it's for
    max_confidence = float(max(probs))
    most_confident_label = int(np.argmax(probs))
    
    # ============================================================
    # STEP 3: SMART DECISION MAKING
    # ============================================================
    
    decision_reason = ""
    
    # CASE 1: BERT is VERY confident (>75%)
    if max_confidence >= BERT_CONFIDENCE_THRESHOLD:
        
        # Trust BERT - it understands semantics better than keyword matching
        final_label = most_confident_label
        final_score = model_score
        
        # But apply sanity checks even for confident BERT
        
        # Sanity Check A: If BERT says Correct but keywords are EXTREMELY low (<10%)
        # AND similarity is also very low (<10%), something is wrong
        if final_label == 2 and kw_coverage < 0.10 and sim < 0.10:
            final_label = 1  # Downgrade to Partial at most
            final_score = min(final_score, 2.5)
            decision_reason = "BERT confident but extremely low concept match - downgraded"
        
        # Sanity Check B: If BERT says Incorrect but keywords are very high (>60%)
        # the model might be wrong
        elif final_label == 0 and kw_coverage > 0.60 and sim > 0.40:
            final_label = 2  # Upgrade to Correct
            final_score = max(final_score, 3.5)
            decision_reason = "BERT said Incorrect but high concept coverage - upgraded"
        
        else:
            decision_reason = f"BERT confident ({max_confidence*100:.0f}%) - trusted model"
    
    # CASE 2: BERT is NOT confident (<75%)
    else:
        # Use rule-based system to help decide
        
        # Sub-case 2a: Both metrics below 30% = INCORRECT
        if kw_coverage < 0.30 and sim < 0.30:
            final_score = max(0.5, min(1.5, (kw_coverage + sim) * 2.5))
            final_label = 0  # Incorrect
            
            if wrong_count > 0:
                final_score = max(0.3, final_score - (wrong_count * 0.3))
            
            decision_reason = "30% Rule: Both keyword and semantic below 30%"
        
        # Sub-case 2b: Good coverage = likely Correct
        elif kw_coverage >= 0.55 and sim >= 0.40:
            base_score = kw_coverage * 5.0 * 0.65 + sim * 5.0 * 0.30 + 0.25
            final_score = max(3.5, min(5.0, base_score))
            final_label = 2  # Correct
            decision_reason = "High keyword and semantic coverage"
        
        # Sub-case 2c: Moderate coverage = Partial
        elif kw_coverage >= 0.30 or sim >= 0.30:
            base_score = kw_coverage * 5.0 * 0.65 + sim * 5.0 * 0.30 + 0.25
            
            if wrong_count > 0:
                base_score -= wrong_count * 0.5
            
            final_score = max(1.5, min(3.5, base_score))
            final_label = 1  # Partial
            decision_reason = "Moderate concept coverage"
        
        # Sub-case 2d: Low coverage = Incorrect
        else:
            base_score = kw_coverage * 5.0 * 0.65 + sim * 5.0 * 0.30
            final_score = max(0.5, min(2.0, base_score))
            final_label = 0  # Incorrect
            decision_reason = "Low concept coverage"
        
        # Additional penalties
        if wrong_count >= 3:
            final_label = min(final_label, 0)
            decision_reason += " + wrong concepts penalty"
    
    # Ensure score matches label
    if final_label == 2 and final_score < 3.0:
        final_score = 3.5
    elif final_label == 0 and final_score > 2.5:
        final_score = min(final_score, 2.0)
    
    final_score = max(0.0, min(5.0, final_score))
    
    # ============================================================
    # STEP 4: EXTRACT CONCEPTS
    # ============================================================
    
    ref_keywords = extract_keywords(reference_answer)
    student_keywords = extract_keywords(student_answer)
    present = ref_keywords & student_keywords
    missing = ref_keywords - student_keywords
    
    # ============================================================
    # STEP 5: FORMAT OUTPUTS
    # ============================================================
    
    label_names = ['❌ Incorrect', '⚠️ Partial', '✅ Correct']
    
    # Score
    score_text = f"## 🎯 Score: {final_score:.2f} / 5.00"
    
    # Label
    label_text = f"## 🏷️ Classification: {label_names[final_label]}"
    label_text += f"\n\n📋 _{decision_reason}_"
    
    if final_label != model_label:
        label_text += f"\n_(Model suggested: {label_names[model_label]} with {max_confidence*100:.0f}% confidence)_"
    
    # Confidence
    confidence_text = f"""📊 **BERT Model Confidence:**

Incorrect: {'█' * int(probs[0] * 30)}{'░' * (30 - int(probs[0] * 30))} {probs[0]*100:.1f}%
Partial:   {'█' * int(probs[1] * 30)}{'░' * (30 - int(probs[1] * 30))} {probs[1]*100:.1f}%
Correct:   {'█' * int(probs[2] * 30)}{'░' * (30 - int(probs[2] * 30))} {probs[2]*100:.1f}%

Confidence Threshold: {BERT_CONFIDENCE_THRESHOLD*100:.0f}%
Model Trusted: {'✅ Yes' if max_confidence >= BERT_CONFIDENCE_THRESHOLD else '❌ No (using rules)'}"""
    
    # Analysis
    kw_status = '✅' if kw_coverage >= 0.50 else '⚠️' if kw_coverage >= 0.30 else '❌'
    sim_status = '✅' if sim >= 0.40 else '⚠️' if sim >= 0.30 else '❌'
    
    analysis_text = f"""### 🔍 Feature Analysis

| Metric | Value | 
|--------|-------|
| **Keyword Coverage** | {kw_coverage*100:.1f}% | 
| **Semantic Similarity** | {sim*100:.1f}% | 
| **Length Ratio** | {length_ratio*100:.1f}% | 

**Word Count:** Student: {student_len} | Reference: {ref_len}
**Wrong Terms:** {wrong_count} {('(' + ', '.join(wrong_words) + ')') if wrong_words else ''}

---
**Decision Logic:**
- BERT Confidence: {max_confidence*100:.0f}% for {['Incorrect', 'Partial', 'Correct'][most_confident_label]}
- {'✅ BERT trusted (high confidence)' if max_confidence >= BERT_CONFIDENCE_THRESHOLD else '📏 Rules applied (low BERT confidence)'}
- {decision_reason}"""
    

# | Metric | Value |
# |--------|-------|
# | **Keyword Coverage** | {kw_coverage*100:.1f}% |
# | **Semantic Similarity** | {sim*100:.1f}% |
# | **Length Ratio** | {length_ratio*100:.1f}% |

# **Word Count:** Student: {student_len} | Reference: {ref_len}
# **Wrong Terms:** {wrong_count} {('(' + ', '.join(wrong_words) + ')') if wrong_words else ''}

# ---
# **Decision Logic:**
# - BERT Confidence: {max_confidence*100:.0f}% for {['Incorrect', 'Partial', 'Correct'][most_confident_label]}
# - {'BERT trusted (high confidence)' if max_confidence >= BERT_CONFIDENCE_THRESHOLD else 'Rules applied (low BERT confidence)'}
# - {decision_reason}"""
    
    # Feedback
    feedback_parts = []
    
    if final_score >= 4.0:
        feedback_parts.append("### 🌟 EXCELLENT!")
        feedback_parts.append("Strong understanding demonstrated.")
    elif final_score >= 2.5:
        feedback_parts.append("### 👍 GOOD EFFORT!")
        feedback_parts.append("Main idea captured, but needs more key concepts.")
    elif final_score >= 1.5:
        feedback_parts.append("### ⚠️ PARTIAL UNDERSTANDING")
        feedback_parts.append("Missing several important concepts.")
    else:
        feedback_parts.append("### 📚 NEEDS SIGNIFICANT WORK")
        feedback_parts.append("The answer misses core concepts or contains errors.")
    
    feedback_parts.append("")
    
    if present:
        feedback_parts.append(f"✅ **Covered:** {', '.join(sorted(list(present))[:8])}")
    else:
        feedback_parts.append("❌ **No key concepts from reference found!**")
    
    feedback_parts.append("")
    
    if missing:
        feedback_parts.append(f"⚠️ **Missing:** {', '.join(sorted(list(missing))[:6])}")
    
    if wrong_words:
        feedback_parts.append(f"❌ **Wrong Terms:** {', '.join(wrong_words)}")
    
    feedback_text = "\n".join(feedback_parts)
    
    # Grok feedback
    print("🤖 Generating AI feedback...")
    
    ai_feedback = generate_grok_feedback(
        question, reference_answer, student_answer,
        final_score, final_label, present, missing, kw_coverage, sim
    )
    
    ai_feedback_text = f"""### 🤖 AI Teacher Feedback

{ai_feedback}"""
    
    return score_text, label_text, confidence_text, analysis_text, feedback_text, ai_feedback_text

# ============================================================
# GRADIO UI
# ============================================================

examples = [
    [
        "What is photosynthesis?",
        "Photosynthesis is the process by which green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen.",
        "Photosynthesis is a process where plants use sunlight, carbon dioxide and water to produce food and release oxygen."
    ],
    [
        "What is machine learning?",
        "Machine learning is a field of artificial intelligence that gives computers the ability to learn from data, identify patterns, and make decisions or predictions without being explicitly programmed for each specific task.",
        "Machine learning is a type of artificial intelligence where you use Python libraries to build a model that predicts things."
    ],
    [
        "What is gradient descent?",
        "Gradient descent is an iterative optimization algorithm used to minimize a function by moving in the direction of steepest descent as defined by the negative of the gradient.",
        "Gradient descent is an optimization technique used in training neural networks to minimize the cost function by adjusting weights based on the gradient."
    ],
    [
        "What is photosynthesis?",
        "Photosynthesis is the process by which green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen.",
        "Animals breathe oxygen and exhale carbon dioxide. This is called respiration. It happens in the lungs."
    ]
]

with gr.Blocks(theme=gr.themes.Soft(), title="AI Answer Grading") as demo:
    
    gr.Markdown("""
    # 🎓 AI-Powered Subjective Answer Grading System
    ### Built with BERT Transformer + Grok AI Feedback
    
    **Model Performance:** QWK 0.77 | Pearson 0.78 | Trained on 6,000+ answers
    
    ---
    **Smart Decision Logic:**
    - 🧠 BERT confidence > 75% → Trust the model (understands paraphrasing)
    - 📏 BERT uncertain → Use keyword/semantic rules
    - ⚠️ Both keywords AND semantics < 30% AND BERT not confident → INCORRECT
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Input")
            
            question_input = gr.Textbox(label="Question", placeholder="Enter the exam question...", lines=3)
            reference_input = gr.Textbox(label="Reference Answer", placeholder="Enter the ideal answer...", lines=5)
            student_input = gr.Textbox(label="Student's Answer", placeholder="Enter student's answer...", lines=5)
            
            grade_btn = gr.Button("🎯 Grade Answer", variant="primary", size="lg")
            
            gr.Markdown("---")
            gr.Examples(examples=examples, inputs=[question_input, reference_input, student_input], label="📖 Examples")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")
            
            score_output = gr.Markdown(label="Score")
            label_output = gr.Markdown(label="Classification")
            
            with gr.Accordion("📈 Model Confidence", open=False):
                confidence_output = gr.Textbox(label="", lines=8, show_label=False)
            
            with gr.Accordion("🔍 Detailed Analysis", open=True):
                analysis_output = gr.Markdown(label="")
            
            with gr.Accordion("💬 Structured Feedback", open=True):
                feedback_output = gr.Markdown(label="")
            
            with gr.Accordion("🤖 AI Teacher Feedback (Grok)", open=True):
                ai_feedback_output = gr.Markdown(label="")
    
    with gr.Accordion("ℹ️ About This System", open=False):
        gr.Markdown("""
        ### Decision Logic
        
        ```
        IF BERT confidence > 75%:
            → TRUST BERT (understands synonyms & paraphrasing)
            → Sanity check: extreme cases still caught
        
        ELSE (BERT uncertain):
            → IF both keyword AND semantic < 30%:
                → INCORRECT (regardless of length!)
            → IF keyword > 55% AND semantic > 40%:
                → CORRECT
            → ELSE:
                → Use weighted features to decide
        ```
        
        ### Why This Works
        - **BERT excels at:** Understanding meaning, synonyms, paraphrasing
        - **Rules excel at:** Catching completely wrong topics, length tricks
        - **Combined:** Best of both worlds!
        """)
    
    grade_btn.click(
        fn=grade_answer,
        inputs=[question_input, reference_input, student_input],
        outputs=[score_output, label_output, confidence_output, analysis_output, feedback_output, ai_feedback_output]
    )

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    print("\n🚀 Launching AI Answer Grading System")
    print("  ✅ BERT model + Rule-based hybrid")
    print("  ✅ Smart confidence thresholding")
    print("  ✅ Grok AI feedback")
    print("  🌐 Generating public URL...\n")
    
    demo.launch(share=True)