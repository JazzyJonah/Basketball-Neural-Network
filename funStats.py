import pandas as pd
import sportsdataverse.mbb as mbb


def load_games(csv_path: str = "mbb_games.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def load_team_lookup(groups: int = 50) -> dict[int, str]:
    teams = mbb.espn_mbb_teams(groups=groups, return_as_pandas=True)
    if not isinstance(teams, pd.DataFrame):
        teams = pd.DataFrame(teams)

    teams = teams[["team_id", "team_display_name"]].dropna().copy()
    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce")
    teams = teams.dropna(subset=["team_id"])
    teams["team_id"] = teams["team_id"].astype(int)

    return dict(zip(teams["team_id"], teams["team_display_name"]))


def team_name(team_id: int, lookup: dict[int, str]) -> str:
    return lookup.get(int(team_id), f"Unknown Team ({team_id})")


def expand_team_games(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "season": row["season"],
                "date": row["date"],
                "team_id": row["team1id"],
                "team_home": row["team1home"],
                "team_pts": row["team1pts"],
                "team_fouls": row["team1pf"],
                "opp_id": row["team2id"],
                "opp_home": row["team2home"],
                "opp_pts": row["team2pts"],
                "opp_fouls": row["team2pf"],
                "point_diff": row["team1pts"] - row["team2pts"],
                "combined_points": row["team1pts"] + row["team2pts"],
                "combined_fouls": row["team1pf"] + row["team2pf"],
            }
        )

        rows.append(
            {
                "season": row["season"],
                "date": row["date"],
                "team_id": row["team2id"],
                "team_home": row["team2home"],
                "team_pts": row["team2pts"],
                "team_fouls": row["team2pf"],
                "opp_id": row["team1id"],
                "opp_home": row["team1home"],
                "opp_pts": row["team1pts"],
                "opp_fouls": row["team1pf"],
                "point_diff": row["team2pts"] - row["team1pts"],
                "combined_points": row["team1pts"] + row["team2pts"],
                "combined_fouls": row["team1pf"] + row["team2pf"],
            }
        )

    return pd.DataFrame(rows)


def _filter_unknown_team_games(
    df: pd.DataFrame, lookup: dict[int, str], ignoreNonD1: bool
) -> pd.DataFrame:
    if not ignoreNonD1:
        return df
    known_ids = set(lookup.keys())
    return df[df["team1id"].isin(known_ids) & df["team2id"].isin(known_ids)].copy()


def print_largest_point_differential(
    df: pd.DataFrame, lookup: dict[int, str], n: int = 5, ignoreNonD1: bool = False
) -> None:
    df = _filter_unknown_team_games(df, lookup, ignoreNonD1)
    team_games = expand_team_games(df)
    top = team_games.sort_values(["point_diff", "date"], ascending=[False, True]).head(
        n
    )

    print("\n=== Largest Point Differential ===")
    for i, row in enumerate(top.itertuples(index=False), start=1):
        print(
            f"{i}. {team_name(row.team_id, lookup)} {row.team_pts} - "
            f"{team_name(row.opp_id, lookup)} {row.opp_pts} | "
            f"Diff: {row.point_diff:+d} | "
            f"Date: {row.date.date()}"
        )


def print_largest_win_ratio(
    df: pd.DataFrame, lookup: dict[int, str], n: int = 5, ignoreNonD1: bool = False
) -> None:
    # Games in which team points / opponent points is greatest
    df = _filter_unknown_team_games(df, lookup, ignoreNonD1)
    team_games = expand_team_games(df)
    team_games["win_ratio"] = team_games["team_pts"] / team_games["opp_pts"].replace(
        0, pd.NA
    )
    top = team_games.sort_values(["win_ratio", "date"], ascending=[False, True]).head(n)

    print("\n=== Largest Win Ratio ===")
    for i, row in enumerate(top.itertuples(index=False), start=1):
        print(
            f"{i}. {team_name(row.team_id, lookup)} {row.team_pts} - "
            f"{team_name(row.opp_id, lookup)} {row.opp_pts} | "
            f"Win Ratio: {row.win_ratio:.2f} | "
            f"Date: {row.date.date()}"
        )


def print_most_total_points(
    df: pd.DataFrame, lookup: dict[int, str], n: int = 5, ignoreNonD1: bool = False
) -> None:
    df = _filter_unknown_team_games(df, lookup, ignoreNonD1)
    top = df.copy()
    top["combined_points"] = top["team1pts"] + top["team2pts"]
    top = top.sort_values(["combined_points", "date"], ascending=[False, True]).head(n)

    print("\n=== Most Total Points Scored ===")
    for i, row in enumerate(top.itertuples(index=False), start=1):
        print(
            f"{i}. {team_name(row.team1id, lookup)} {row.team1pts} - "
            f"{team_name(row.team2id, lookup)} {row.team2pts} | "
            f"Combined: {row.combined_points} | "
            f"Date: {row.date.date()}"
        )


def print_fewest_team_fouls(
    df: pd.DataFrame, lookup: dict[int, str], n: int = 5, ignoreNonD1: bool = False
) -> None:
    df = _filter_unknown_team_games(df, lookup, ignoreNonD1)
    team_games = expand_team_games(df)
    top = team_games.sort_values(["team_fouls", "date"], ascending=[True, True]).head(n)

    print("\n=== Fewest Team Fouls ===")
    for i, row in enumerate(top.itertuples(index=False), start=1):
        print(
            f"{i}. {team_name(row.team_id, lookup)} vs {team_name(row.opp_id, lookup)} | "
            f"Fouls: {row.team_fouls} | "
            f"Score: {row.team_pts}-{row.opp_pts} | "
            f"Date: {row.date.date()}"
        )


if __name__ == "__main__":
    games = load_games("mbb_games.csv")
    lookup = load_team_lookup()

    # print_largest_point_differential(games, lookup, n=5, ignoreNonD1=True)
    # print_largest_win_ratio(games, lookup, n=50, ignoreNonD1=True)
    # print_most_total_points(games, lookup, n=5, ignoreNonD1=True)
    # print_fewest_team_fouls(games, lookup, n=5, ignoreNonD1=True)
