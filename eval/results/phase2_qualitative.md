# Phase 2 — qualitative cross-modal retrieval examples

Backend: **openclip-vit-h14** · gallery: 1000 images (Flickr30k 1K test gallery). Quantitative headline: R@5 = 0.942 (`phase3b_headline.md`). Image files are git-ignored; results shown by `image_id` + representative caption so relevance is judgeable from text alone.

## Text → image (caption query, cross-modal)

**Query: "a dog running on the beach"**

  1. `11618` (score 0.346) — A gray and white dog jumping over standing water in the sand.
  2. `5441` (score 0.325) — A small black and white dog running on the beach with several people in the background
  3. `9736` (score 0.316) — A black dog with a blue tag running on a beach
  4. `1833` (score 0.309) — Light brown dog running towards something at the beach.
  5. `17822` (score 0.304) — A brown dog is shown standing in the water near a muddy beach.

**Query: "two people riding bicycles in a city"**

  1. `22098` (score 0.259) — On a clear day, a number of cyclists ride their bikes over a bridge in an industrial area.
  2. `24425` (score 0.245) — A man in a black hat and jacket is riding his bicycle.
  3. `24315` (score 0.237) — A little boy with a blue bike helmet and a woman in a green bike helmet are riding bikes down a street with other various bike riders.
  4. `29952` (score 0.231) — Two women sit on a brown scooter, one is wearing blue jeans and a brown jacket, the other is carrying a bag and has on black pants and a black top with a green hat.
  5. `3087` (score 0.229) — A girl sits on a decorated bike with a younger boy while another girl takes a picture.

**Query: "a child playing with a ball in a park"**

  1. `23065` (score 0.351) — A small child in a blue outfit is looking off at some trees in the distance.
  2. `9598` (score 0.317) — A little girl with a brown shirt, blue jeans, and brown hair is playing outside with her pink scooter in a yard with lots of trees.
  3. `1390` (score 0.295) — A young man wearing blue jeans and a t-shirt sits in the grass, with a ball in the air.
  4. `10124` (score 0.267) — A boy in a stripy shirt is playing with a soccer ball in a grassy park in front of some bushes.
  5. `23724` (score 0.251) — A boy wearing shorts, crocs, and a white shirt plays with a yo-yo while another boy looks at his yo-yo in the background.

**Query: "a person climbing a snowy mountain"**

  1. `17630` (score 0.283) — A lone person jumping through the air from one snowy mountain to another.
  2. `25986` (score 0.269) — A snowboarder is catching air on a snowy hill while wearing protective snow gear.
  3. `28246` (score 0.252) — A woman smiles as she skis, wearing a white winter hat, a white coat, and a blue hoodie, while snow is in the background.
  4. `2906` (score 0.245) — Climbers with hiking boots and blue helmets ascend a snow covered mountain.
  5. `2909` (score 0.211) — A mountaineer about to descend down a mountain with a blue helmet on.

**Query: "a group of friends eating at a restaurant"**

  1. `19027` (score 0.249) — A group of people eats in yellow chairs at a restaurant, with many fans and lights on the ceiling.
  2. `24145` (score 0.247) — Two little boys eat McDonald's while sitting at a table and chairs outside, where others are also eating.
  3. `25069` (score 0.243) — A group of men gathered around a table preparing to play a card game.
  4. `10693` (score 0.233) — A woman holding a red cup with a straw in it sits in front of a man and a woman looking at a book.
  5. `9328` (score 0.225) — A group of young people take shots in a Mexican setting.

**Query: "a man playing an electric guitar on stage"**

  1. `25819` (score 0.304) — A man in a white shirt playing a white guitar standing in front of a microphone with the light casting a somewhat reddish hue on his shirt
  2. `29480` (score 0.257) — Three musicians on a stage, two are guitarists, and one is wearing a hat, black striped pants, strumming a red electric guitar, and the other guitarist is strumming a white guitar while the other musician is wearing blue jeans and a black t-shirt and singing.
  3. `9042` (score 0.254) — A band performing on stage while the crowd observes with there hands in the air.
  4. `15309` (score 0.239) — A long-haired, male musician is playing on a piano.
  5. `1984` (score 0.224) — A girl is playing an electric guitar in front of an amplifier.

## Image → image (reverse image search)

**Query image: `10005`** (top hit should be itself, score ≈ 1.0)

  1. `10005` (score 1.000) — A child in a orange shirt is pouring Legos out of a plastic bag.
  2. `4942` (score 0.637) — A small boy plays with plastic blocks, cars, and animals with an adult watching closely.
  3. `8740` (score 0.635) — A woman sits at a primary colored children's table, playing with building blocks, as two girls in pink dresses, and a boy in a red shirt surround her.
  4. `4686` (score 0.610) — A little boy holding a yellow, plastic shovel, crouches in the sand.

**Query image: `10010`** (top hit should be itself, score ≈ 1.0)

  1. `10010` (score 1.000) — A woman is holding her child while a man is reading something from a piece of paper and taking something from a bowl that a younger girl is holding in a park with a small lake in the background.
  2. `14626` (score 0.572) — Men and women sitting and walking around picnic tables and having food.
  3. `8631` (score 0.534) — A bald man holds a fish in front of a lake while two blond young children stand near him while holding fishing poles.
  4. `434` (score 0.486) — A group of people standing on the lawn in front of a building.

**Query image: `10088`** (top hit should be itself, score ≈ 1.0)

  1. `10088` (score 1.000) — Two men and a woman pose with signs for Obama and Chris Gregoire.
  2. `10858` (score 0.414) — A group of people at a protest to stop the building of the Kingsworth coal power plant.
  3. `10106` (score 0.387) — A couple of men in hats posing for a picture by a gas station.
  4. `17933` (score 0.370) — People are gathered around the table filled with food.
