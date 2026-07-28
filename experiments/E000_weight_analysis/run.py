from genweight import ModelLoader, WeightStatistics, WeightVisualizer


def main():

    loader = ModelLoader("gpt2")

    loader.load()

    weight = loader.get_parameter(
        "h.0.attn.c_attn.weight"
    )

    stats = WeightStatistics(weight)

    summary = stats.summary()

    print()

    print("=" * 60)

    print("Weight Statistics")

    print("=" * 60)

    for key, value in summary.items():

        print(f"{key:<20} : {value}")

    # Visualization
    visualizer = WeightVisualizer(weight)

    visualizer.histogram(
        bins=100,
        save_path="results/weight_distribution.png",
    )

    visualizer.heatmap(
        save_path="results/weight_heatmap.png",
    )


if __name__ == "__main__":
    main()