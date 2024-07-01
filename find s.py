def find_s(training_data):
    # Initialize the most specific hypothesis
    hypothesis = ['0'] * len(training_data[0])  # Start with the most specific hypothesis
    
    for instance in training_data:
        # Only consider positive instances (assuming the class label is at the end)
        if instance[-1] == 'Yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':  # Hypothesis is most specific, generalize it
                    hypothesis[i] = instance[i]
                elif hypothesis[i] != instance[i]:  # If there is a conflict, generalize it
                    hypothesis[i] = '?'
    
    return hypothesis

# Example training data [attributes..., class_label]
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Running FIND-S algorithm on the training data
hypothesis = find_s(training_data)
print("The most specific hypothesis found by FIND-S algorithm is:")
print(hypothesis)
 
