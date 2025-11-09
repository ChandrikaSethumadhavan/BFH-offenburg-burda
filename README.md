# Meals or Miles: Recipe Carbon Footprint Analyzer



Prototype with PyQt:

https://github.com/user-attachments/assets/48e07576-747c-44e5-95b3-76ff6d786dbb



> How Much Does Your Food Really Emit?

**Black Forest Hackathon, Offenburg 2025**



## The Problem

Ever wondered if your weekly meal emissions compete with Taylor Swift's jet? We help you find out! Most people care about the environment but don't know which ingredients in their favorite recipes contribute most to carbon emissions. Our tool bridges this knowledge gap by analyzing recipes and suggesting sustainable alternatives without sacrificing nutrition.

## What It Does?

**Meals or Miles** analyzes your recipes to:
- Calculate the total CO₂ footprint of your meal
- Identify key emission contributors (ingredients >15% of total CO₂)
- Suggest sustainable ingredient swaps
- Maintain similar nutritional profiles (protein, carbs, fat)
- Visualize real-time CO₂ reduction comparisons with real time comparisons

### Example Use Case
Monkey wants to make classic Goulash with minced beef but cares about the environment. Our app suggests replacing minced beef with soybeans, showing a dramatic reduction in carbon emissions while maintaining nutritional balance.





## Key Features

- **Lightweight & Fast** - Reduced computation time, minimal algorithmic carbon footprint
- **No API Dependency** - Works offline with local database
- **Multi-language Support** - Analyze recipes in multiple languages
- **Repeatable** - Consistent results every time
- **User-Friendly Interface** - Simple, intuitive design

## Tech Stack

- Vector embeddings for ingredient matching
- Nutritional database integration
- Real-time visualization engine
- Multi-objective optimization algorithms


## Future Roadmap

- [ ] Expand database with curated food-based carbon emissions data
- [ ] Increase parameters for multi-objective optimization
- [ ] Recipe-level reinforcement optimization learning from user swaps
- [ ] User personalization with likes/dislikes filters
- [ ] Community recipe sharing


## Getting Started

```bash
# Clone the repository
git clone https://github.com/randomwordgenerator/meals-or-miles.git

# Navigate to project directory
cd meals-or-miles

# Install dependencies
npm install

# Run the application
npm start
```



##  Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


##  Team

- Muthu Sai Charan, FAU Erlangen-Nurnberg
- Chandrika Sethumadhavan, University of Freiburg
- Edwin, University of Offenburg

##  Acknowledgments

- Black Forest Hackathon, Offenburg 2025
- Burda Solutions
- University of Koblenz for the emissions dataset
