import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    df = pl.read_csv("data/accepted_2007_to_2018Q4.csv", ignore_errors=True).filter(pl.col("id").is_not_null())

    df.head()
    return


if __name__ == "__main__":
    app.run()
