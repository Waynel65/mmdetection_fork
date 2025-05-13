def generate_cosine_schedulers(max_epochs, period_length=10, warmup_length=1, warmup_ratio=0.1):
    """
    Generate a list of scheduler dictionaries with the following pattern:
    1. CosineAnnealingLR for epochs 0-9, 11-19, 21-29, etc.
    2. LinearLR warmup at epochs 10, 20, 30, etc. to bring LR back up
    
    Args:
        max_epochs (int): Total number of epochs for training
        period_length (int): Length of each full cycle (cosine + warmup) in epochs
        warmup_length (int): Length of each warmup phase in epochs (typically 1)
        warmup_ratio (float): Start factor for warmup phases (0.1 means start at 10% of the base LR)
        
    Returns:
        list: List of scheduler dictionaries
    """
    schedulers = []
    current_epoch = 0
    
    while current_epoch < max_epochs:
        # Determine if this is a warmup point (multiple of 10)
        is_warmup_point = current_epoch > 0 and current_epoch % period_length == 0
        
        if is_warmup_point and current_epoch + warmup_length <= max_epochs:
            # Add a LinearLR warmup at epochs 10, 20, 30, etc.
            warmup = dict(
                type="LinearLR",
                start_factor=warmup_ratio,  # Start at X% of base LR
                by_epoch=True,
                begin=current_epoch,
                end=current_epoch + warmup_length
            )
            schedulers.append(warmup)
            current_epoch += warmup_length
        else:
            # Determine end of current cosine phase
            # It should end right before the next multiple of 10, or at max_epochs
            next_warmup_point = ((current_epoch // period_length) + 1) * period_length
            cosine_end = min(next_warmup_point, max_epochs)
            
            if current_epoch < cosine_end:
                # Add a cosine annealing phase
                cosine = dict(
                    type="CosineAnnealingLR",
                    T_max=cosine_end - current_epoch,  # Dynamic T_max
                    by_epoch=True,
                    begin=current_epoch,
                    end=cosine_end
                )
                schedulers.append(cosine)
            
            current_epoch = cosine_end
    

    print(schedulers)
    return schedulers