import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
BUCKET_WIDTH = 0.05
EDGES = np.arange(0.0, 1.0 + BUCKET_WIDTH, BUCKET_WIDTH)  # 0.00 ... 1.00
N_BUCKETS = len(EDGES) - 1  # 20
ROUNDS = 5
TOSSES_PER_ROUND = 5
POINTS_TOTAL = 100


def bucket_labels():
    labels = []
    for i in range(N_BUCKETS):
        lo = EDGES[i]
        hi = EDGES[i + 1]
        # last bucket inclusive at 1.00
        if i == N_BUCKETS - 1:
            labels.append(f"{lo:.2f}-{hi:.2f}")
        else:
            labels.append(f"{lo:.2f}-{hi:.2f}")
    return labels


def bucket_index_for_p(p: float) -> int:
    """
    Returns bucket index for p in [0,1].
    Buckets are [0.00,0.05), [0.05,0.10), ..., [0.95,1.00] (last inclusive).
    """
    if p >= 1.0:
        return N_BUCKETS - 1
    if p <= 0.0:
        return 0
    idx = int(p // BUCKET_WIDTH)
    return min(max(idx, 0), N_BUCKETS - 1)


def parse_points_input(raw: str) -> np.ndarray:
    """
    Accepts either:
      - 20 integers separated by commas/spaces
      - OR a compact form like: "0:10,1:5,2:0,..."
    but simplest is 20 numbers.
    """
    raw = raw.strip()
    if ":" in raw:
        # sparse format: "idx:points, idx:points, ..."
        pts = np.zeros(N_BUCKETS, dtype=int)
        chunks = [c.strip() for c in raw.split(",") if c.strip()]
        for ch in chunks:
            idx_str, val_str = ch.split(":")
            idx = int(idx_str.strip())
            val = int(val_str.strip())
            if not (0 <= idx < N_BUCKETS):
                raise ValueError(f"Bucket index {idx} out of range 0..{N_BUCKETS-1}")
            pts[idx] = val
        return pts

    # dense format: 20 numbers
    parts = raw.replace(",", " ").split()
    if len(parts) != N_BUCKETS:
        raise ValueError(f"Expected {N_BUCKETS} numbers, got {len(parts)}.")
    pts = np.array([int(x) for x in parts], dtype=int)
    return pts


def prompt_for_distribution(round_name: str) -> np.ndarray:
    labels = bucket_labels()
    print("\nAllocate 100 points across these 20 probability buckets (sum must be 100):")
    print("Buckets (index: range):")
    for i, lab in enumerate(labels):
        print(f"  {i:2d}: {lab}")

    print("\nInput format options:")
    print("  A) 20 integers (recommended), e.g.:")
    print("     0 0 0 0 5 10 15 20 20 15 10 5 0 0 0 0 0 0 0 0")
    print("  B) Comma-separated also works.")
    print("  C) Sparse format: idx:points,idx:points,... (unspecified buckets assumed 0)\n")

    while True:
        raw = input(f"{round_name} - enter your {POINTS_TOTAL} points: ")
        try:
            pts = parse_points_input(raw)
            if np.any(pts < 0):
                raise ValueError("Points cannot be negative.")
            s = int(pts.sum())
            if s != POINTS_TOTAL:
                raise ValueError(f"Points must sum to {POINTS_TOTAL}; got {s}.")
            return pts
        except Exception as e:
            print(f"Invalid input: {e}\nPlease try again.\n")


def plot_distribution(points: np.ndarray, title: str):
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    plt.figure(figsize=(10, 4))
    plt.bar(centers, points, width=BUCKET_WIDTH * 0.95)
    plt.xticks(EDGES, rotation=90)
    plt.xlabel("Coin head probability bucket")
    plt.ylabel("Points")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def final_plot(all_distributions, p_true: float):
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    plt.figure(figsize=(10, 4))

    # plot each distribution in red (semi-transparent)
    for k, pts in enumerate(all_distributions, start=1):
        plt.plot(centers, pts, marker="o", linewidth=1, alpha=0.6)  # default color is fine, but request says red
    # enforce red explicitly for "all the others are red"
    # (Matplotlib defaults vary; we set them to red to match your spec.)
    plt.clf()
    plt.figure(figsize=(10, 4))
    for pts in all_distributions:
        plt.plot(centers, pts, marker="o", linewidth=1, alpha=0.6, color="red")

    # true probability in green
    plt.axvline(p_true, linewidth=2, color="green")

    plt.xticks(EDGES, rotation=90)
    plt.xlabel("Coin head probability bucket")
    plt.ylabel("Points")
    plt.title("All guesses (red) vs true p (green)")
    plt.tight_layout()
    plt.show()


def run_coin_bayes_game(p_true=None, seed=None):
    rng = np.random.default_rng(seed)

    # choose true p if not provided
    if p_true is None:
        # keep away from exact 0/1 to avoid triviality
        p_true = float(rng.uniform(0.05, 0.95))

    print("Coin Toss Belief Game")
    print("---------------------")
    print(f"(Hidden) True head probability has been set.\n")

    cumulative_heads = 0
    cumulative_tails = 0

    distributions = []

    for r in range(1, ROUNDS + 1):
        # simulate 5 tosses
        tosses = rng.random(TOSSES_PER_ROUND) < p_true
        heads = int(tosses.sum())
        tails = TOSSES_PER_ROUND - heads

        cumulative_heads += heads
        cumulative_tails += tails

        print(f"\nRound {r} results (5 tosses): Heads={heads}, Tails={tails}")
        print(f"Cumulative after {r*TOSSES_PER_ROUND} tosses: Heads={cumulative_heads}, Tails={cumulative_tails}")

        pts = prompt_for_distribution(round_name=f"Round {r}")
        distributions.append(pts)
        plot_distribution(pts, title=f"Your distribution after round {r} ({r*TOSSES_PER_ROUND} tosses)")

    # final guess after 25 tosses
    final_pts = prompt_for_distribution(round_name="FINAL GUESS (after 25 tosses)")
    distributions.append(final_pts)
    plot_distribution(final_pts, title="Your FINAL distribution (used for payout)")

    # payout
    true_bucket = bucket_index_for_p(p_true)
    points_in_true_bucket = int(final_pts[true_bucket])
    money_won = points_in_true_bucket * 5

    # reveal truth + final plot
    print("\n--- REVEAL ---")
    print(f"True p(heads) = {p_true:.4f}")
    print(f"True bucket index = {true_bucket}  (range {EDGES[true_bucket]:.2f}-{EDGES[true_bucket+1]:.2f})")
    print(f"Points you placed in the true bucket (FINAL) = {points_in_true_bucket}")
    print(f"Money won = {points_in_true_bucket} x 5 = {money_won}")

    final_plot(distributions, p_true=p_true)

    return {
        "p_true": p_true,
        "true_bucket": true_bucket,
        "money_won": money_won,
        "distributions": distributions,
        "cumulative_heads": cumulative_heads,
        "cumulative_tails": cumulative_tails,
    }


# Example run:
# results = run_coin_bayes_game(seed=42)
# print(results["money_won"])