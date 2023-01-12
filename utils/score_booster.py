def cos_sim_score_with_threshold(score, eps, alpha, threshold):
    
    '''        
        Gets the score identifies to increase or decrease it and returns new update score.
        
        Arguments:
        
        score - score within range of 0~1;
        eps - epsilon value, hyperparameter;
        alpha - alpha value, hyperparameter;
        threshold - value to classify the score as positive or negative.
    '''   
    
    # Show initial score
    print(f"Original score: {score}")
    if score >= threshold:
        return (score + eps) / (eps + alpha)
    elif score < threshold:
        return abs((score + (alpha / eps)) / (2*eps))

def cos_sim_score_booster(score, eps, alpha, mode):
    
    '''        
        Gets the score and mode and returns new update score based on the mode.
        
        Arguments:
        
        score - score within range of 0~1;
        eps - epsilon value, hyperparameter;
        alpha - alpha value, hyperparameter;
        mode - mode to increase or decrease the score.
    '''   
    
    if mode == "for_pos":
        return (score + eps) / (eps + alpha)
    elif mode == "for_neg":
        return abs((score + (alpha / eps)) / (2*eps))
