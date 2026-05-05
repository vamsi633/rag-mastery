import json
from config.settings import client,LLM_MODEL

system_prompt="""Grade if this document chunk helps answer the question.

-"relevant":Directly helps answer the question
-"irrelevant":Does not help at all
-"ambiguous": Tangentially related but may not fully answer

Be STRICT. Respond with JSON: {"grade": "relevant/irrelevant/ambiguous", "reason": "brief"}
"""
def grade_chunk(query,chunk):
    response=client.chat.completions.create(
        model=LLM_MODEL,messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"Question: {query}\n\nChunk:\n{chunk['text']}"}
        ],
        temperature=0,
        response_format={"type":"json_object"},
    )
    return json.loads(response.choices[0].message.content)

def grade_and_filter(query,chunks):
    """
    Grade all chunks and decide what to do.
    Returns: (action, filtered_chunks)

    acrion: "CORRECT" | "AMBIGUOUS" | "INCORRECT"
    """

    graded=[]

    for chunk in chunks:
        result=grade_chunk(query,chunk)
        chunk["grade"]=result.get("grade","ambiguous")
        chunk["grade_reason"]=result.get("reason","")
        graded.append(chunk)

    relevant=[c for c in graded if c["grade"]=="relevant"]
    ambiguous=[c for c in graded if c["grade"]=="ambiguous"]

    if len(relevant) >= 2:
        return "CORRECT", relevant, graded
    elif len(relevant) >= 1 or len(ambiguous) >= 2:
        return "AMBIGUOUS", relevant + ambiguous, graded
    else:
        return "INCORRECT", [], graded
