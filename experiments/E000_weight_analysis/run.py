from genweight import ModelLoader


def main():

    loader = ModelLoader("gpt2")

    loader.load()

    print("\nTotal Parameters")

    print(f"{loader.parameter_count():,}\n")

    print("Listing Parameters...\n")

    loader.list_parameters()

    weight = loader.get_parameter(
        "h.0.attn.c_attn.weight"
    )

    print("\nSelected Matrix")

    print(weight.shape)

    print(weight.dtype)


if __name__ == "__main__":
    main()