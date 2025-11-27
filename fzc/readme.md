使用示例如下：

```python
from poly_score import infer_scores

request_score, response_score = infer_scores(
    "The capital of France is Paris.",
    "Yes, that is correct."
)

# request_score = infer_scores("The capital of France is Paris.")
print(request_score)
print(response_score)
