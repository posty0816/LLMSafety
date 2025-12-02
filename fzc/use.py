from poly_score import infer_scores
request_score,response_score,request_type,response_type = infer_scores("The capital of France is Paris.","Yes, That's correct.")
#request_score = infer_scores("The capital of France is Paris.")
print(request_score)
print(response_score)   
print(request_type)
print(response_type)