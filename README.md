<p align="center">
    <strong>Enhancing movielens data with IMDb metadata</strong>
</p>
This repository uses the [imdbpy.github.io](https://imdbpy.github.io/) to fetch the metadata for movielens movies. The movielens dataset contains
a csv file that has the mapping of movielens id to IMDb id. This id is used to fetch the main attributes with IMDbPY.
Since IMDbPY does not fetch all attributes I employ Beautiful Soup to fetch additional metadata, such as:

- Stars (main actors of the movie)
- Demographic data (age & gender)
- Distribution of ratings
- US Users and Non-US Users

For the movie avatar this can be found here:
- [imdb.com/title/tt0499549/?ref_=nv_sr_srsg_3](https://www.imdb.com/title/tt0499549/?ref_=nv_sr_srsg_3)
- [imdb.com/title/tt0499549/ratings](https://www.imdb.com/title/tt0499549/ratings)


The mapping is available in data/input/movielens/small/links.

Excerpt:

| movieId | imdbId  | tmdbId  |
|---|---|---|
| 1  | 0114709  |  862 |
|  2 |  0113497 | 8844  |
|  3 | 0113228  | 15602  |

## Keywords
- 🎥 MovieLens
- 👥 IMDb Metadata

## Architecture
- Python 3.7
- IMDbPY
- Beautifoul Soup 4 for metadata that is not available by using pyIMDb

## ToDo
- Create bin size for continous features
