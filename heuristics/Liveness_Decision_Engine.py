from typing import Dict, List

def compute_final_verdict(results: Dict) -> Dict:
    """
    Aggregates all modalities into a final pass/fail decision.
    results: { 'rppg': bpm, 'moire': bool, 'sensor': status, '3d': bool, 'texture': count, 'gaze': dir }
    """
    confidence = 0.0
    fail_reasons = []

    # 1. Injection Check (CRITICAL)
    if results.get("sensor") == "VIRTUAL":
        return {"verdict": "FAILED", "confidence": 1.0, "reason": "Virtual Camera / Injection Detected"}

    # 2. Anti-Spoofing (Passive)
    if results.get("moire"):
        fail_reasons.append("Screen artifacts detected (Moire)")
    else:
        confidence += 0.25

    # 3. 3D Depth Check
    if results.get("is_3d"):
        confidence += 0.25
    else:
        fail_reasons.append("Surface appears 2D / Flat")

    # 4. Skin Texture
    if results.get("spots", 0) > 5:
        confidence += 0.25
    else:
        fail_reasons.append("Skin texture too smooth (suspected screen/mask)")

    # 5. Physiological (rPPG)
    if results.get("bpm", 0) > 40: # Valid heart rate detected
        confidence += 0.25
    
    # Verdict Logic
    verdict = "PASSED" if confidence >= 0.75 and not results.get("moire") else "SUSPICIOUS"
    if not results.get("is_3d") and results.get("moire"):
        verdict = "FAILED"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasons": fail_reasons
    }
