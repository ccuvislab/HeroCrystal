from pt.engine.FLtrainer import FLtrainer

def get_match_array_nogt_batch(proposals_roih):
    # Create an instance of FLtrainer
    trainer = FLtrainer()
    # Call the get_match_array_nogt_batch method
    result = trainer.get_match_array_nogt_batch(proposals_roih)
    return result


def main():
    # Read input from text file
    input_file = 'output_images/bbox_to_match_0_batch_size_2.txt'

    with open(input_file, "r") as f:
        bbox_to_match = [line.strip() for line in f]

    

    print(bbox_to_match)

    # Call get_match_array_nogt_batch with the input
    result = get_match_array_nogt_batch(bbox_to_match)

    # Print the result
    print(result)

main()