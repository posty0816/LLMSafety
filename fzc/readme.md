使用示例如下：
(在输出类别时，如果无害输出0，如果有害只输出最可能违反的类别)

```python
from poly_score import infer_scores
request_score,response_score,request_type,response_type = infer_scores("The capital of France is Paris.","Yes, That's correct.")
#request_score = infer_scores("The capital of France is Paris.")
print(request_score)
print(response_score)   
print(request_type)
print(response_type)
```
类别：
- **S1: Violent Crimes**
- **S2: Non-Violent Crimes**
- **S3: Sex Crimes**
- **S4: Child Exploitation**
- **S5: Defamation**
- **S6: Specialized Advice**
- **S7: Privacy**
- **S8: Intellectual Property**
- **S9: Indiscriminate Weapons**
- **S10: Hate**
- **S11: Self-Harm**
- **S12: Sexual Content**
- **S13: Elections**
- **S14: Code Interpreter Abuse**

