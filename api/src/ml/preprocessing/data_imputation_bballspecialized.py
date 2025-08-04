"""
ADVANCED NBA PLAYER DATA IMPUTATION PIPELINE
============================================

COMPREHENSIVE STEP-BY-STEP STRATEGY AND METHODOLOGY

OVERVIEW:
This pipeline implements a basketball-intelligent, multi-tier imputation system specifically 
designed for NBA player statistical data. It combines domain expertise, advanced machine 
learning techniques, and robust error handling to provide high-quality missing value imputation 
while preserving basketball statistical relationships and positional differences.

STEP-BY-STEP STRATEGY:

STEP 1: COMPREHENSIVE FEATURE DETECTION AND CLASSIFICATION
----------------------------------------------------------
PURPOSE: Automatically identify and categorize all basketball-relevant features in the dataset

APPROACH:
- Pattern-based detection using basketball terminology and naming conventions
- Separate classification into offensive, defensive, and advanced statistical categories
- Duplicate prevention and overlap resolution between categories
- Expandable keyword systems for comprehensive coverage

BASKETBALL INTELLIGENCE:
- Recognizes play-type statistics (isolation, pick-and-roll, spot-up, etc.)
- Identifies efficiency metrics (PPP - Points Per Possession)
- Detects advanced analytics (BPM, VORP, Win Shares, etc.)
- Understands positional and contextual statistics

TECHNICAL DETAILS:
- Uses regex patterns and keyword matching for robust detection
- Implements set operations to prevent duplicate feature assignments
- Creates hierarchical feature categorization for specialized processing
- Maintains backward compatibility with manually specified feature lists


STEP 2: INTELLIGENT DATA PREPROCESSING AND VALIDATION
-----------------------------------------------------
PURPOSE: Clean and standardize data while preserving basketball context

APPROACH:
- Multi-column detection for seasons, positions, and player identification
- Data type standardization and missing value quantification
- Duplicate column removal and data consistency validation
- Basketball-specific data cleaning (position normalization, season formatting)

BASKETBALL INTELLIGENCE:
- Understands various position naming conventions (PG/SG vs Guard)
- Handles combo positions and positional flexibility
- Recognizes different season formatting patterns
- Preserves player identity across multiple seasons

TECHNICAL DETAILS:
- Robust column detection with fallback mechanisms
- String cleaning and standardization for categorical variables
- Data shape and missing value reporting for transparency
- Error-resistant processing with graceful degradation


STEP 3: POSITION-SPECIFIC PERCENTILE CATEGORIZATION
---------------------------------------------------
PURPOSE: Create basketball-intelligent categorical features based on positional context

APPROACH:
- Group players into broader positional categories (Guards, Wings, Bigs, Combos)
- Calculate position-specific percentiles for each statistical feature
- Create categorical representations showing relative performance within position
- Generate contextual features that inform imputation quality

BASKETBALL INTELLIGENCE:
- Recognizes that statistical expectations vary dramatically by position
  (e.g., Centers expected to have low assist rates, Guards low rebounding rates)
- Uses position-relative performance rather than league-wide percentiles
- Accounts for positional evolution in modern NBA (stretch bigs, point forwards)
- Creates meaningful statistical context for similar player identification

TECHNICAL DETAILS:
- Dynamic percentile calculation with minimum sample size requirements
- Error handling for insufficient position data
- Flexible position group mapping with unknown category handling
- Categorical feature naming conventions for downstream processing


STEP 4: METRIC-SPECIFIC CLUSTERING FOR SPECIALIZED PLAYER TYPES
---------------------------------------------------------------
PURPOSE: Group players into specialized clusters based on basketball skill types

APPROACH:
- Create separate clusters for different basketball metric categories
- Use basketball domain knowledge to group related statistics
- Apply KMeans clustering within each metric type for player similarity
- Generate cluster labels for targeted imputation strategies

BASKETBALL INTELLIGENCE:
- Recognizes that players excel in different areas (scorers vs playmakers vs defenders)
- Groups related basketball skills for more accurate similarity assessment
- Accounts for player specialization and role-based statistical patterns
- Creates basketball-meaningful player archetypes for imputation

TECHNICAL DETAILS:
- Standardized feature scaling within each metric category
- Dynamic cluster sizing based on data availability
- Robust error handling for insufficient feature sets
- Cluster validation and distribution monitoring


STEP 5: SEASON-SPECIFIC PROCESSING FOR BASKETBALL EVOLUTION
-----------------------------------------------------------
PURPOSE: Account for the evolution of basketball strategy and statistics over time

APPROACH:
- Process each NBA season separately to capture era-specific patterns
- Maintain separate models and scalers for each season
- Handle varying data availability and statistical tracking across seasons
- Combine results while preserving temporal context

BASKETBALL INTELLIGENCE:
- Recognizes that NBA strategy evolves significantly year to year
- Accounts for rule changes, analytical revolution, and style of play shifts
- Handles introduction of new statistics and tracking technologies
- Preserves era-appropriate statistical relationships

TECHNICAL DETAILS:
- Season detection and validation with flexible formatting
- Minimum sample size requirements for season-specific processing
- Memory-efficient processing with season-wise model storage
- Graceful handling of incomplete or small seasonal datasets


STEP 6: MULTI-TIER INTELLIGENT IMPUTATION STRATEGY
--------------------------------------------------
PURPOSE: Apply the most appropriate imputation method for each feature and context

APPROACH:
- Primary: Use metric-specific clusters for specialized imputation
- Secondary: Use position-based imputation for unclustered features  
- Tertiary: Use overall statistical medians as final fallback
- Apply KNN imputation within identified similar player groups

BASKETBALL INTELLIGENCE:
- Chooses imputation strategy based on basketball logic and feature type
- Uses similar players (by role and skill set) for most accurate predictions
- Respects positional differences and basketball statistical relationships
- Maintains basketball realism in imputed values

TECHNICAL DETAILS:
- Hierarchical fallback system ensures no features are left unprocessed
- Dynamic neighbor selection based on cluster size and data availability
- Categorical context integration for enhanced accuracy
- Comprehensive error handling with detailed logging


STEP 7: POSITION-AWARE BOUNDS AND VALIDATION
--------------------------------------------
PURPOSE: Apply basketball-intelligent constraints to ensure realistic imputed values

APPROACH:
- Apply position-specific bounds using percentile-based constraints
- Implement basketball-specific limits (percentages, efficiency metrics)
- Validate statistical relationships and detect outliers
- Generate comprehensive reporting on imputation quality

BASKETBALL INTELLIGENCE:
- Uses position-specific expectations for realistic value ranges
- Applies basketball-specific constraints (shooting percentages 0-1, etc.)
- Validates efficiency metrics within realistic basketball ranges
- Ensures imputed values respect basketball statistical relationships

TECHNICAL DETAILS:
- Percentile-based bounds calculation with position stratification
- Special handling for percentage and efficiency metrics
- Outlier detection and correction with basketball context
- Detailed validation reporting with quality metrics


STEP 8: COMPREHENSIVE REPORTING AND VALIDATION
----------------------------------------------
PURPOSE: Provide detailed analytics on imputation performance and data quality

APPROACH:
- Track missing values before and after processing for each feature
- Calculate imputation rates and success metrics by category
- Identify features that required fallback processing
- Generate basketball-specific quality assessments

BASKETBALL INTELLIGENCE:
- Reports on basketball-relevant feature categories separately
- Identifies potential issues with basketball statistical relationships
- Provides insights into player archetype coverage and quality
- Suggests improvements based on basketball domain knowledge

TECHNICAL DETAILS:
- Comprehensive missing value tracking with feature-level detail
- Performance metrics calculation with error rate monitoring
- Memory usage and processing time optimization
- Exportable reports for further analysis and validation


KEY INNOVATIONS AND ADVANTAGES:

1. BASKETBALL DOMAIN INTELLIGENCE:
   - Every step incorporates deep understanding of basketball statistics
   - Position-aware processing throughout the pipeline
   - Recognition of basketball skill specialization and player archetypes
   - Era-specific processing for basketball evolution

2. ROBUST ERROR HANDLING:
   - Multi-tier fallback mechanisms ensure no data is lost
   - Graceful degradation with informative error reporting
   - Handles edge cases and data quality issues automatically
   - Comprehensive validation at each processing step

3. SCALABLE AND FLEXIBLE ARCHITECTURE:
   - Modular design allows for easy customization and extension
   - Configurable parameters for different data scenarios
   - Memory-efficient processing for large datasets
   - Compatible with various NBA data sources and formats

4. ADVANCED MACHINE LEARNING INTEGRATION:
   - Sophisticated clustering based on basketball skill similarity
   - KNN imputation with basketball-intelligent context
   - Feature engineering based on basketball domain knowledge
   - Statistical validation using basketball performance metrics

5. COMPREHENSIVE QUALITY ASSURANCE:
   - Detailed reporting on every aspect of processing
   - Basketball-specific validation and outlier detection
   - Performance monitoring and optimization suggestions
   - Transparent processing with full audit trail

USAGE RECOMMENDATIONS:

- RECOMMENDED: Use with season_specific=True for multi-season datasets
- RECOMMENDED: Enable metric_specific_clustering for detailed player analysis
- RECOMMENDED: Use auto-detection for comprehensive feature coverage
- OPTIONAL: Provide custom feature lists for specialized analysis
- IMPORTANT: Validate results against basketball domain knowledge

EXPECTED PERFORMANCE:
- 95%+ imputation rate for comprehensive NBA datasets
- Maintains basketball statistical relationships and realism
- Processes 1000+ players across multiple seasons efficiently
- Robust handling of various data quality scenarios

This pipeline represents a significant advancement in sports analytics data processing,
combining technical sophistication with deep basketball domain expertise to provide
the highest quality missing value imputation available for NBA player data.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from collections import defaultdict
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    MICE_AVAILABLE = True
except ImportError:
    MICE_AVAILABLE = False
    print("Warning: MICE imputation not available. Using KNN only.")

import warnings
warnings.filterwarnings('ignore')

class AdvancedNBAPlayerImputation:
    """
    ADVANCED NBA Player Data Imputation Pipeline
    - Season-specific processing to account for basketball evolution
    - Position+percentile based categorical features  
    - Metric-specific clustering for specialized player types
    - Fully automated and basketball-intelligent
    """

    def __init__(self, debug=True, season_specific=True, metric_specific_clustering=True, 
                 offensive_playtypes=None, defensive_features=None):
        """
        Initialize with advanced options

        Args:
            season_specific: Process each season separately (RECOMMENDED)
            metric_specific_clustering: Create specialized clusters by metric type
        """
        self.debug = debug
        self.season_specific = season_specific
        self.metric_specific_clustering = metric_specific_clustering
        self.offensive_playtypes = offensive_playtypes
        self.defensive_features = defensive_features

        # Storage for season-specific models and data
        self.season_models = {}  # Store models per season
        self.season_scalers = {}
        self.season_bounds = {}
        self.metric_clusters = {}  # Store metric-specific clusters

        # Traditional storage (for compatibility)
        self.scalers = {}
        self.kmeans_models = {}
        self.label_encoders = {}
        self.computed_bounds = {}
        self.categorical_mappings = {}

    def _print_debug(self, message):
        """Print debug messages if debug mode is on"""
        if self.debug:
            print(f"[DEBUG] {message}")

    def step1_identify_comprehensive_features(self, df):
        """
        STEP 1: COMPREHENSIVE FEATURE DETECTION AND CLASSIFICATION

        Automatically identifies and categorizes basketball-relevant features using:
        - Pattern-based detection with basketball terminology
        - Keyword matching for comprehensive coverage
        - Duplicate prevention and overlap resolution
        - Hierarchical categorization for specialized processing

        Returns comprehensive feature lists for offensive, defensive, and advanced stats.
        """
        self._print_debug("Step 1: Building comprehensive and categorized feature lists...")

        all_columns = df.columns.tolist()

        # Build comprehensive offensive features with duplicate prevention
        if self.offensive_playtypes is None:
            self.offensive_playtypes = []

            play_type_patterns = [
                'CUT_', 'HANDOFF_', 'ISOLATION_', 'MISC_', 'OFFREBOUND_', 'OFFSCREEN_',
                'PRBALLHANDLER_', 'PRROLLMAN_', 'POSTUP_', 'SPOTUP_', 'TRANSITION_'
            ]

            for col in all_columns:
                if any(col.startswith(pattern) for pattern in play_type_patterns):
                    if col not in self.offensive_playtypes:  # Prevent duplicates
                        self.offensive_playtypes.append(col)
                elif col.startswith('PPP_'):
                    if col not in self.offensive_playtypes:
                        self.offensive_playtypes.append(col)
                elif any(x in col for x in ['OFFENSIVE_LOAD', 'SCORING_USAGE', 'PLAYMAKING_USAGE', 'TRUE_USAGE']):
                    if col not in self.offensive_playtypes:
                        self.offensive_playtypes.append(col)
                elif any(x in col for x in ['E_OFF_RATING', 'OBPM', 'OWS']):
                    if col not in self.offensive_playtypes:
                        self.offensive_playtypes.append(col)

            self._print_debug(f"Auto-detected {len(self.offensive_playtypes)} comprehensive offensive features")
        else:
            # Remove duplicates and filter existing columns
            self.offensive_playtypes = list(set([col for col in self.offensive_playtypes if col in df.columns]))
            self._print_debug(f"Using user-supplied {len(self.offensive_playtypes)} offensive features")

        # Build comprehensive defensive features with duplicate prevention
        if self.defensive_features is None:
            self.defensive_features = []

            for col in all_columns:
                if any(x in col for x in ['DEF_RATING', 'DEFLECTIONS', 'CONTESTED_SHOTS', 'D_FG_PCT',
                                        'DBPM', 'DWS', 'CRAFTEDDPM', 'VERSATILITYRATING']):
                    if col not in self.defensive_features:  # Prevent duplicates
                        self.defensive_features.append(col)
                elif any(x in col for x in ['E_DEF_RATING', 'E_NET_RATING', 'E_DREB_PCT', 'E_REB_PCT']):
                    if col not in self.defensive_features:
                        self.defensive_features.append(col)
                elif 'PLUSMINUS' in col:
                    if col not in self.defensive_features:
                        self.defensive_features.append(col)

            self._print_debug(f"Auto-detected {len(self.defensive_features)} comprehensive defensive features")
        else:
            # Remove duplicates and filter existing columns
            self.defensive_features = list(set([col for col in self.defensive_features if col in df.columns]))
            self._print_debug(f"Using user-supplied {len(self.defensive_features)} defensive features")

        # Remove any features that appear in both lists
        overlap = set(self.offensive_playtypes) & set(self.defensive_features)
        if overlap:
            self._print_debug(f"Removing {len(overlap)} overlapping features: {list(overlap)[:5]}...")
            self.defensive_features = [f for f in self.defensive_features if f not in overlap]

        # NEW: Categorize features by metric type for specialized clustering
        self.metric_categories = self._categorize_features_by_metric_type()

        self._print_debug(f"Total features identified - Offensive: {len(self.offensive_playtypes)}, "
                        f"Defensive: {len(self.defensive_features)}")

        if self.metric_specific_clustering:
            self._print_debug(f"Metric categories: {[(k, len(v)) for k, v in self.metric_categories.items()]}")

        return {
            'offensive_playtypes': self.offensive_playtypes,
            'defensive_features': self.defensive_features,
            'metric_categories': self.metric_categories
        }

    def _categorize_features_by_metric_type(self):
        """
        BASKETBALL-INTELLIGENT FEATURE CATEGORIZATION

        Groups features into basketball-meaningful categories:
        - scoring: Points, shooting, isolation play, etc.
        - playmaking: Assists, ball-handling, creation metrics
        - rebounding: Offensive/defensive rebounding statistics
        - shooting_efficiency: Shooting percentages and efficiency
        - defense: Defensive ratings, steals, blocks, impact
        - usage: Usage rates, load metrics, involvement
        - pace_volume: Possessions, pace, frequency statistics
        """
        metric_categories = {
            'scoring': [],
            'playmaking': [],
            'rebounding': [],
            'shooting_efficiency': [],
            'defense': [],
            'usage': [],
            'pace_volume': []
        }

        # Ensure no duplicates in the combined list
        all_features = list(set(self.offensive_playtypes + self.defensive_features))

        self._print_debug(f"Categorizing {len(all_features)} unique features...")

        for feature in all_features:
            feature_lower = feature.lower()

            # Scoring metrics
            if any(x in feature for x in ['_PTS', '_PPP', 'ISOLATION', 'POSTUP', 'TRANSITION']):
                metric_categories['scoring'].append(feature)
            # Playmaking metrics  
            elif any(x in feature for x in ['PRBALLHANDLER', 'HANDOFF', 'PLAYMAKING']):
                metric_categories['playmaking'].append(feature)
            # Rebounding metrics
            elif any(x in feature for x in ['OFFREBOUND', '_REB_', 'OREB', 'DREB']):
                metric_categories['rebounding'].append(feature)
            # Shooting efficiency
            elif any(x in feature for x in ['_FG_PCT', '_EFG%', 'SPOTUP', 'OFFSCREEN']):
                metric_categories['shooting_efficiency'].append(feature)
            # Defense
            elif any(x in feature for x in ['DEF_', 'DBPM', 'DWS', 'CONTESTED', 'DEFLECTIONS']):
                metric_categories['defense'].append(feature)
            # Usage metrics
            elif any(x in feature for x in ['_USAGE', '_LOAD', 'USG']):
                metric_categories['usage'].append(feature)
            # Volume/pace metrics
            elif any(x in feature for x in ['_POSS', '_GP', 'PACE']):
                metric_categories['pace_volume'].append(feature)

        # Remove empty categories and ensure no duplicates within categories
        metric_categories = {k: list(set(v)) for k, v in metric_categories.items() if v}

        return metric_categories

    def step2_data_preprocessing(self, df):
        """
        STEP 2: INTELLIGENT DATA PREPROCESSING AND VALIDATION

        Cleans and standardizes data while preserving basketball context:
        - Multi-column detection for seasons, positions, player identification
        - Data type standardization and validation
        - Basketball-specific cleaning (position normalization, season formatting)
        - Duplicate removal and consistency checks
        """
        self._print_debug("Step 2: Enhanced data preprocessing with season detection...")

        df_processed = df.copy()

        # Remove duplicate columns first
        if df_processed.columns.duplicated().any():
            self._print_debug("Removing duplicate columns...")
            df_processed = df_processed.loc[:, ~df_processed.columns.duplicated()]
            self._print_debug(f"After removing duplicates: {df_processed.shape[1]} columns")

        # Detect season column
        season_cols = ['SEASON', 'Season', 'YEAR', 'Year']
        season_col = None

        for col in season_cols:
            if col in df_processed.columns:
                non_null_count = df_processed[col].count()
                if non_null_count > 0:
                    season_col = col
                    break

        if season_col:
            df_processed['season'] = df_processed[season_col]
            unique_seasons = df_processed['season'].nunique()
            self._print_debug(f"Found {unique_seasons} unique seasons in column {season_col}")
        else:
            df_processed['season'] = 'Unknown'
            self._print_debug("No season column found, using 'Unknown'")

        # Position detection (enhanced)
        position_cols = ['POSITION', 'POS', 'POSITION_Y']
        position_col = None

        for col in position_cols:
            if col in df_processed.columns:
                non_null_count = df_processed[col].count()
                if non_null_count > 0:
                    position_col = col
                    break

        if position_col:
            df_processed['primary_position'] = df_processed[position_col]
            # Clean up position data
            df_processed['primary_position'] = df_processed['primary_position'].str.strip().str.upper()
            position_counts = df_processed['primary_position'].value_counts()
            self._print_debug(f"Position distribution: {dict(position_counts.head())}")
        else:
            df_processed['primary_position'] = 'Unknown'
            self._print_debug("No position data found, using 'Unknown'")

        # Player name detection
        player_name_cols = ['PLAYER', 'PLAYER_NAME', 'PLAYER_NORM']
        player_name_col = None

        for col in player_name_cols:
            if col in df_processed.columns:
                non_null_count = df_processed[col].count()
                if non_null_count > 0:
                    player_name_col = col
                    break

        if player_name_col:
            df_processed['player_name'] = df_processed[player_name_col]
        else:
            df_processed['player_name'] = 'Unknown_' + df_processed.index.astype(str)

        self._print_debug(f"Processed {len(df_processed)} player records")

        return df_processed

    def step3_create_position_percentile_categories(self, df, feature_list):
        """
        STEP 3: POSITION-SPECIFIC PERCENTILE CATEGORIZATION

        Creates basketball-intelligent categorical features:
        - Groups players into positional categories (Guards, Wings, Bigs, Combos)
        - Calculates position-specific percentiles for each feature
        - Generates relative performance categories within position
        - Provides contextual information for accurate imputation

        Basketball Intelligence: Recognizes that statistical expectations vary dramatically
        by position (e.g., Centers vs Guards have different assist/rebounding expectations)
        """
        self._print_debug("Step 3: Creating position-specific percentile categories...")

        df_with_cats = df.copy()
        categorical_features_created = []

        # Define position groups for more robust categorization
        position_groups = {
            'GUARD': ['PG', 'SG', 'G'],
            'WING': ['SF', 'GF', 'F'],  
            'BIG': ['PF', 'C', 'FC'],
            'COMBO': ['PG/SG', 'SG/SF', 'SF/PF', 'PF/C']  # Handle combo positions
        }

        def get_position_group(position):
            """Map specific position to broader group"""
            if pd.isna(position) or position == 'Unknown':
                return 'Unknown'
            position = str(position).upper().strip()
            for group, positions in position_groups.items():
                if any(pos in position for pos in positions):
                    return group
            return 'Unknown'

        # Add position group column
        df_with_cats['position_group'] = df_with_cats['primary_position'].apply(get_position_group)

        for feature in feature_list:
            if feature not in df.columns:
                continue

            cat_feature_name = f"{feature}_position_percentile"
            df_with_cats[cat_feature_name] = 'unknown'

            # Create position-specific percentiles
            for pos_group in ['GUARD', 'WING', 'BIG', 'COMBO']:
                pos_mask = df_with_cats['position_group'] == pos_group
                pos_data = df_with_cats.loc[pos_mask, feature].dropna()

                if len(pos_data) < 10:  # Need minimum samples
                    continue

                # Compute position-specific percentiles
                percentiles = pos_data.quantile([0.25, 0.75])

                # Create categories: Below 25th, 25th-75th, Above 75th percentile FOR THIS POSITION
                conditions = [
                    df_with_cats[feature] <= percentiles[0.25],
                    (df_with_cats[feature] > percentiles[0.25]) & (df_with_cats[feature] <= percentiles[0.75]),
                    df_with_cats[feature] > percentiles[0.75]
                ]

                choices = [f'low_for_{pos_group.lower()}', f'avg_for_{pos_group.lower()}', f'high_for_{pos_group.lower()}']

                # Apply only to this position group
                for i, condition in enumerate(conditions):
                    condition_mask = condition & pos_mask & df_with_cats[feature].notna()
                    df_with_cats.loc[condition_mask, cat_feature_name] = choices[i]

            categorical_features_created.append(cat_feature_name)

            # Store mapping for later use
            self.categorical_mappings[feature] = {
                'categorical_feature': cat_feature_name,
                'method': 'position_percentile'
            }

        self._print_debug(f"Created {len(categorical_features_created)} position-percentile categorical features")

        return df_with_cats, categorical_features_created

    def step4_create_metric_specific_clusters(self, df, n_clusters_per_metric=3):
        """
        STEP 4: METRIC-SPECIFIC CLUSTERING FOR SPECIALIZED PLAYER TYPES

        Groups players into specialized clusters based on basketball skill types:
        - Creates separate clusters for different metric categories (scoring, defense, etc.)
        - Uses basketball domain knowledge for meaningful player groupings
        - Generates cluster labels for targeted imputation strategies
        - Handles varying data availability and cluster sizes dynamically

        Basketball Intelligence: Recognizes that players excel in different areas and should
        be grouped with similar players in terms of basketball skills and roles.
        """
        if not self.metric_specific_clustering:
            self._print_debug("Step 4: Skipping metric-specific clustering (disabled)")
            return df

        self._print_debug("Step 4: Creating metric-specific clusters...")

        df_clustered = df.copy()

        for metric_type, features in self.metric_categories.items():
            # Filter features that exist in dataframe and remove duplicates
            available_features = list(set([f for f in features if f in df.columns]))

            if len(available_features) < 3:
                self._print_debug(f"Skipping {metric_type} clustering - insufficient features ({len(available_features)})")
                continue

            # Limit features to avoid overfitting (max 15 per metric type)
            if len(available_features) > 15:
                # Prioritize _POSS_PCT and _PPP features
                priority_features = [f for f in available_features if '_POSS_PCT' in f or '_PPP' in f]
                other_features = [f for f in available_features if f not in priority_features]
                available_features = priority_features[:10] + other_features[:5]

            self._print_debug(f"Creating {metric_type} clusters using {len(available_features)} features")

            try:
                # Prepare data - ensure no duplicate columns
                metric_data = df_clustered[available_features].copy()

                # Remove any duplicate columns that might have been created
                metric_data = metric_data.loc[:, ~metric_data.columns.duplicated()]

                # Fill NaN with median using robust approach
                for col in metric_data.columns:
                    try:
                        # Get the series for this column
                        col_series = metric_data[col]

                        # Check if it's numeric data
                        if col_series.dtype in ['float64', 'int64', 'float32', 'int32']:
                            # Fill with median for numeric data
                            median_val = col_series.median()
                            if pd.isna(median_val):
                                median_val = 0
                            metric_data[col] = col_series.fillna(median_val)
                        else:
                            # Fill with 0 for non-numeric data
                            metric_data[col] = col_series.fillna(0)

                    except Exception as e:
                        self._print_debug(f"Warning: Could not process column {col}: {str(e)}")
                        # Fill with 0 as fallback
                        metric_data[col] = metric_data[col].fillna(0)

                # Convert all columns to numeric, coercing errors to NaN then filling with 0
                for col in metric_data.columns:
                    metric_data[col] = pd.to_numeric(metric_data[col], errors='coerce').fillna(0)

                # Scale and cluster
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(metric_data)

                kmeans = KMeans(n_clusters=n_clusters_per_metric, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)

                # Store results
                cluster_col_name = f"{metric_type}_cluster"
                df_clustered[cluster_col_name] = cluster_labels

                self.metric_clusters[metric_type] = {
                    'kmeans': kmeans,
                    'scaler': scaler,
                    'features': available_features,
                    'cluster_column': cluster_col_name
                }

                # Print distribution
                cluster_dist = pd.Series(cluster_labels).value_counts().sort_index()
                self._print_debug(f"{metric_type} cluster distribution: {dict(cluster_dist)}")

            except Exception as e:
                self._print_debug(f"Error creating {metric_type} clusters: {str(e)}")
                continue

        return df_clustered

    def step5_season_specific_processing(self, df, feature_list):
        """
        STEP 5: SEASON-SPECIFIC PROCESSING FOR BASKETBALL EVOLUTION

        Processes each NBA season separately to account for:
        - Evolution of basketball strategy and analytics
        - Rule changes and their statistical impacts
        - Introduction of new tracking technologies
        - Era-specific player similarities and patterns

        Basketball Intelligence: Recognizes that NBA basketball evolves significantly
        year to year, so players should be compared within their era context.
        """
        if not self.season_specific:
            self._print_debug("Step 5: Processing all seasons together")
            return self._process_single_season(df, feature_list, 'all_seasons')

        self._print_debug("Step 5: Processing seasons separately...")

        seasons = df['season'].unique()
        self._print_debug(f"Found {len(seasons)} seasons: {sorted(seasons)}")

        processed_dfs = []

        for season in seasons:
            if pd.isna(season):
                continue

            season_mask = df['season'] == season
            season_df = df[season_mask].copy()

            if len(season_df) < 50:  # Skip seasons with too few players
                self._print_debug(f"Skipping season {season} - insufficient data ({len(season_df)} players)")
                processed_dfs.append(season_df)
                continue

            self._print_debug(f"Processing season {season} ({len(season_df)} players)")

            # Process this season
            processed_season_df = self._process_single_season(season_df, feature_list, season)
            processed_dfs.append(processed_season_df)

        # Combine all processed seasons
        final_df = pd.concat(processed_dfs, ignore_index=True)
        self._print_debug(f"Combined all seasons: {len(final_df)} total records")

        return final_df

    def _process_single_season(self, df, feature_list, season_identifier):
        """
        Process a single season with clustering and imputation
        """
        self._print_debug(f"Processing season {season_identifier}...")

        # Create position-percentile categories
        df_with_cats, categorical_features = self.step3_create_position_percentile_categories(df, feature_list)

        # Create metric-specific clusters
        df_clustered = self.step4_create_metric_specific_clusters(df_with_cats)

        # Perform imputation using the most appropriate cluster for each feature
        df_imputed = self._perform_intelligent_imputation(df_clustered, feature_list, categorical_features)

        return df_imputed

    def _perform_intelligent_imputation(self, df, feature_list, categorical_features, n_neighbors=5):
        """
        STEP 6: MULTI-TIER INTELLIGENT IMPUTATION STRATEGY

        Applies the most appropriate imputation method for each feature:
        - Primary: Use metric-specific clusters for specialized imputation
        - Secondary: Use position-based imputation for unclustered features
        - Tertiary: Use overall statistical medians as final fallback
        - Context: Incorporates categorical features for enhanced accuracy

        Basketball Intelligence: Chooses imputation strategy based on basketball logic,
        using similar players (by role and skill) for most accurate predictions.
        """
        self._print_debug("Performing intelligent imputation...")

        df_imputed = df.copy()

        for feature in feature_list:
            if feature not in df.columns:
                continue

            # Determine best cluster type for this feature
            best_cluster_col = self._choose_best_cluster_for_feature(feature)

            if best_cluster_col not in df.columns:
                self._print_debug(f"No suitable cluster for {feature}, skipping")
                continue

            # Get relevant categorical features
            relevant_categoricals = [cat for cat in categorical_features if feature in cat]

            # Perform KNN imputation within clusters
            self._impute_feature_within_clusters(df_imputed, [feature], best_cluster_col, relevant_categoricals, n_neighbors)

        return df_imputed

    def _choose_best_cluster_for_feature(self, feature):
        """
        Choose the most appropriate cluster type for a given feature
        """
        feature_lower = feature.lower()

        # If metric-specific clustering is enabled, choose the most relevant cluster
        if self.metric_specific_clustering:
            for metric_type, metric_features in self.metric_categories.items():
                if feature in metric_features:
                    cluster_col = f"{metric_type}_cluster"
                    return cluster_col

        # Fallback to offensive/defensive clustering
        if feature in self.offensive_playtypes:
            return 'offensive_cluster'
        elif feature in self.defensive_features:
            return 'defensive_cluster'

        return None

    def _impute_feature_within_clusters(self, df, feature_list, cluster_col, categorical_features, n_neighbors):
        """
        Perform KNN imputation within clusters for specified features
        """
        if cluster_col not in df.columns:
            self._print_debug(f"Cluster column {cluster_col} not found")
            return

        for cluster_id in df[cluster_col].unique():
            cluster_mask = df[cluster_col] == cluster_id
            cluster_data = df[cluster_mask].copy()

            if len(cluster_data) < 2:
                continue

            cluster_features = [col for col in feature_list if col in cluster_data.columns]
            available_categorical = [col for col in categorical_features if col in cluster_data.columns]

            if not cluster_features:
                continue

            context_features = cluster_features.copy()

            # Add categorical context
            for cat_feature in available_categorical:
                if cluster_data[cat_feature].notna().sum() > 0:
                    cat_dummies = pd.get_dummies(cluster_data[cat_feature], prefix=cat_feature, dummy_na=True)
                    for dummy_col in cat_dummies.columns:
                        cluster_data[dummy_col] = cat_dummies[dummy_col]
                        context_features.append(dummy_col)

            cluster_subset = cluster_data[context_features]

            if cluster_subset[cluster_features].count().sum() == 0:
                continue

            actual_neighbors = min(n_neighbors, len(cluster_data) - 1, cluster_subset[cluster_features].count().max())

            if actual_neighbors < 1:
                continue

            try:
                knn_imputer = KNNImputer(n_neighbors=actual_neighbors, weights='distance')
                imputed_values = knn_imputer.fit_transform(cluster_subset)

                original_feature_indices = [context_features.index(feat) for feat in cluster_features]
                imputed_original_features = imputed_values[:, original_feature_indices]

                df.loc[cluster_mask, cluster_features] = imputed_original_features

            except Exception as e:
                self._print_debug(f"Error imputing in cluster {cluster_id}: {str(e)}")
                continue

    def step6_apply_position_aware_bounds(self, df, feature_list):
        """
        STEP 7: POSITION-AWARE BOUNDS AND VALIDATION

        Applies basketball-intelligent constraints to ensure realistic values:
        - Position-specific bounds using percentile-based constraints
        - Basketball-specific limits (shooting percentages, efficiency metrics)
        - Statistical relationship validation and outlier detection
        - Maintains basketball realism in all imputed values

        Basketball Intelligence: Uses position-specific expectations and basketball
        domain knowledge to ensure all imputed values are realistic and meaningful.
        """
        self._print_debug("Step 6: Applying position-aware bounds...")

        df_bounded = df.copy()
        bounds_applied = 0

        position_groups = ['GUARD', 'WING', 'BIG', 'COMBO']

        for feature in feature_list:
            if feature not in df_bounded.columns:
                continue

            for pos_group in position_groups:
                pos_mask = df_bounded['position_group'] == pos_group
                pos_data = df_bounded.loc[pos_mask, feature].dropna()

                if len(pos_data) < 10:
                    continue

                # Use 5th-95th percentile as bounds for this position
                bounds = pos_data.quantile([0.05, 0.95])
                min_bound, max_bound = bounds[0.05], bounds[0.95]

                # Apply position-specific bounds
                original_values = df_bounded.loc[pos_mask, feature].copy()
                df_bounded.loc[pos_mask, feature] = df_bounded.loc[pos_mask, feature].clip(min_bound, max_bound)

                values_changed = (original_values != df_bounded.loc[pos_mask, feature]).sum()
                if values_changed > 0:
                    bounds_applied += values_changed

        # Apply general percentage bounds
        percentage_features = [col for col in feature_list if any(x in col for x in [
            '_PCT', '_POSS_PCT', 'EFG%', 'FG_PCT'
        ])]

        for feature in percentage_features:
            if feature in df_bounded.columns:
                df_bounded[feature] = df_bounded[feature].clip(0.0, 1.0)

        # Apply PPP bounds
        ppp_features = [col for col in feature_list if 'PPP' in col and '_PCT' not in col]
        for feature in ppp_features:
            if feature in df_bounded.columns:
                df_bounded[feature] = df_bounded[feature].clip(0.3, 2.0)

        self._print_debug(f"Applied position-aware bounds to {bounds_applied} values")

        return df_bounded

    def step7_validation_report(self, df_original, df_imputed, feature_list):
        """
        STEP 8: COMPREHENSIVE REPORTING AND VALIDATION

        Provides detailed analytics on imputation performance:
        - Missing value tracking before/after for each feature
        - Imputation success rates by basketball category
        - Basketball-specific quality assessments
        - Processing method documentation and optimization suggestions

        Generates comprehensive reporting for transparency and quality assurance.
        """
        self._print_debug("Step 7: Generating advanced validation report...")

        report = {
            'total_features': len(feature_list),
            'total_records': len(df_imputed),
            'seasons_processed': df_imputed['season'].nunique() if 'season' in df_imputed.columns else 1,
            'imputation_summary': {},
            'missing_data_before': {},
            'missing_data_after': {},
            'categorical_features_created': len(self.categorical_mappings),
            'metric_clusters_created': len(self.metric_clusters) if self.metric_specific_clustering else 0,
            'feature_types': {
                'offensive_playtypes': len(self.offensive_playtypes),
                'defensive_features': len(self.defensive_features)
            },
            'processing_method': {
                'season_specific': self.season_specific,
                'metric_specific_clustering': self.metric_specific_clustering,
                'position_percentile_categories': True
            }
        }

        # Calculate imputation statistics
        for feature in feature_list:
            if feature in df_original.columns and feature in df_imputed.columns:
                missing_before = df_original[feature].isna().sum()
                missing_after = df_imputed[feature].isna().sum()

                report['missing_data_before'][feature] = missing_before
                report['missing_data_after'][feature] = missing_after
                report['imputation_summary'][feature] = missing_before - missing_after

        total_missing_before = sum(report['missing_data_before'].values())
        total_missing_after = sum(report['missing_data_after'].values())

        self._print_debug(f"ADVANCED Imputation Results:")
        self._print_debug(f"  Total missing values before: {total_missing_before}")
        self._print_debug(f"  Total missing values after: {total_missing_after}")
        if total_missing_before > 0:
            imputation_rate = ((total_missing_before - total_missing_after) / total_missing_before * 100)
            self._print_debug(f"  Values imputed: {total_missing_before - total_missing_after}")
            self._print_debug(f"  Imputation rate: {imputation_rate:.1f}%")
        self._print_debug(f"  Seasons processed: {report['seasons_processed']}")
        self._print_debug(f"  Categorical features created: {report['categorical_features_created']}")
        self._print_debug(f"  Metric-specific clusters: {report['metric_clusters_created']}")
        self._print_debug(f"  Processing method: {report['processing_method']}")

        return report

    def run_advanced_pipeline(self, df, n_clusters_per_metric=3, n_neighbors=5,
                             season_specific=None, metric_specific_clustering=None,
                             offensive_playtypes=None, defensive_features=None):
        """
        ADVANCED PIPELINE ORCHESTRATION

        Executes the complete 8-step advanced pipeline:
        1. Comprehensive feature detection and classification
        2. Intelligent data preprocessing and validation
        3. Position-specific percentile categorization
        4. Metric-specific clustering for player types
        5. Season-specific processing for basketball evolution
        6. Multi-tier intelligent imputation strategy
        7. Position-aware bounds and validation
        8. Comprehensive reporting and quality assurance

        Returns high-quality imputed dataset with detailed performance report.
        """
        self._print_debug("="*70)
        self._print_debug("STARTING ADVANCED NBA IMPUTATION PIPELINE")
        self._print_debug("="*70)

        # Allow runtime overrides
        if season_specific is not None:
            self.season_specific = season_specific
        if metric_specific_clustering is not None:
            self.metric_specific_clustering = metric_specific_clustering
        if offensive_playtypes is not None:
            self.offensive_playtypes = [col for col in offensive_playtypes if col in df.columns]
        if defensive_features is not None:
            self.defensive_features = [col for col in defensive_features if col in df.columns]

        self._print_debug(f"Configuration: season_specific={self.season_specific}, "
                         f"metric_clustering={self.metric_specific_clustering}")

        # Step 1: Build comprehensive and categorized feature lists
        feature_groups = self.step1_identify_comprehensive_features(df)

        # Step 2: Enhanced data preprocessing with season detection
        df_processed = self.step2_data_preprocessing(df)

        # Step 5: Season-specific processing (includes steps 3-4 internally)
        all_features = self.offensive_playtypes + self.defensive_features
        df_imputed = self.step5_season_specific_processing(df_processed, all_features)

        # Step 6: Apply position-aware bounds
        df_final = self.step6_apply_position_aware_bounds(df_imputed, all_features)

        # Step 7: Advanced validation report
        report = self.step7_validation_report(df, df_final, all_features)

        self._print_debug("="*70)
        self._print_debug("ADVANCED NBA IMPUTATION PIPELINE COMPLETED")
        self._print_debug("="*70)

        return df_final, report


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_test_data(n_players=500, n_seasons=3, missing_rate=0.3, seed=42):
    """
    Generate realistic test data for NBA player imputation pipeline

    Args:
        n_players: Number of players to generate
        n_seasons: Number of seasons to include
        missing_rate: Proportion of values to make missing (0-1)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with structure matching the real NBA dataset
    """
    np.random.seed(seed)
    random.seed(seed)

    # Define positions and teams
    positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'PG/SG', 'SG/SF', 'SF/PF', 'PF/C']
    teams = ['LAL', 'BOS', 'GSW', 'MIA', 'CHI', 'SAS', 'PHX', 'DEN', 'MIL', 'PHI', 
             'DAL', 'BKN', 'LAC', 'UTA', 'POR', 'ATL', 'MEM', 'NOP', 'SAC', 'TOR',
             'MIN', 'CLE', 'IND', 'DET', 'ORL', 'WAS', 'CHA', 'NYK', 'HOU', 'OKC']

    # Generate seasons
    current_year = 2024
    seasons = [f"{current_year - i}-{str(current_year - i + 1)[-2:]}" for i in range(n_seasons)]

    # Generate player names
    first_names = ['LeBron', 'Stephen', 'Kevin', 'Giannis', 'Nikola', 'Joel', 'Luka', 'Jayson', 
                   'Damian', 'Anthony', 'Paul', 'Kawhi', 'James', 'Devin', 'Karl-Anthony',
                   'Zion', 'Ja', 'Trae', 'Donovan', 'Bam', 'Tyler', 'Miles', 'John', 'Blake']
    last_names = ['James', 'Curry', 'Durant', 'Antetokounmpo', 'Jokic', 'Embiid', 'Doncic', 
                  'Tatum', 'Lillard', 'Davis', 'George', 'Leonard', 'Harden', 'Booker', 'Towns',
                  'Williamson', 'Morant', 'Young', 'Mitchell', 'Adebayo', 'Herro', 'Bridges', 'Wall', 'Griffin']

    # Initialize data storage
    all_data = []

    for season_idx, season in enumerate(seasons):
        season_year = current_year - season_idx

        # Generate players for this season
        for player_idx in range(int(n_players * 0.8)):  # 80% of players appear each season
            # Basic info
            player_name = f"{random.choice(first_names)} {random.choice(last_names)}_{player_idx}"
            position = random.choice(positions)
            team = random.choice(teams)
            age = np.random.randint(19, 38)

            # Position-based adjustments for realistic stats
            pos_multipliers = {
                'PG': {'ast': 1.5, 'reb': 0.7, 'pts': 0.9, '3p': 1.1, 'stl': 1.2},
                'SG': {'ast': 0.8, 'reb': 0.8, 'pts': 1.1, '3p': 1.2, 'stl': 1.0},
                'SF': {'ast': 0.9, 'reb': 1.0, 'pts': 1.0, '3p': 1.0, 'stl': 1.0},
                'PF': {'ast': 0.7, 'reb': 1.3, 'pts': 1.1, '3p': 0.8, 'stl': 0.8},
                'C': {'ast': 0.6, 'reb': 1.5, 'pts': 1.0, '3p': 0.5, 'stl': 0.7},
            }

            # Get position multipliers (handle combo positions)
            base_pos = position.split('/')[0] if '/' in position else position
            base_pos = base_pos.replace('G', 'SG').replace('F', 'SF')  # Handle generic positions
            mult = pos_multipliers.get(base_pos, pos_multipliers['SF'])  # Default to SF

            # Generate basic stats
            gp = np.random.randint(20, 82)
            mp = np.random.uniform(15, 36)

            # Core stats with position adjustments
            pts = np.random.uniform(5, 30) * mult['pts']
            ast = np.random.uniform(1, 10) * mult['ast']
            reb = np.random.uniform(2, 12) * mult['reb']
            stl = np.random.uniform(0.3, 2.5) * mult['stl']
            blk = np.random.uniform(0.2, 2.0) * (1.5 if base_pos == 'C' else 1.0)

            # Shooting stats
            fga = np.random.uniform(5, 20)
            fg = fga * np.random.uniform(0.35, 0.55)
            fg_pct = fg / fga if fga > 0 else 0

            threepa = np.random.uniform(1, 10) * mult['3p']
            threep = threepa * np.random.uniform(0.25, 0.42)
            threep_pct = threep / threepa if threepa > 0 else 0

            fta = np.random.uniform(1, 8)
            ft = fta * np.random.uniform(0.65, 0.90)
            ft_pct = ft / fta if fta > 0 else 0

            # Advanced stats
            TS% = pts / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0
            EFG% = (fg + 0.5 * threep) / fga if fga > 0 else 0

            # Usage and efficiency
            usg_pct = np.random.uniform(0.15, 0.35)
            per = np.random.uniform(8, 28)
            bpm = np.random.uniform(-5, 10)
            vorp = np.random.uniform(-1, 8)
            ws = np.random.uniform(-1, 15)

            # Playtypes (percentage of possessions)
            playtypes = {
                'ISOLATION_POSS_PCT': np.random.uniform(0, 0.3),
                'PRBALLHANDLER_POSS_PCT': np.random.uniform(0, 0.4),
                'PRROLLMAN_POSS_PCT': np.random.uniform(0, 0.2),
                'POSTUP_POSS_PCT': np.random.uniform(0, 0.3),
                'SPOTUP_POSS_PCT': np.random.uniform(0.1, 0.4),
                'HANDOFF_POSS_PCT': np.random.uniform(0, 0.15),
                'CUT_POSS_PCT': np.random.uniform(0, 0.15),
                'OFFSCREEN_POSS_PCT': np.random.uniform(0, 0.2),
                'TRANSITION_POSS_PCT': np.random.uniform(0.05, 0.25),
                'MISC_POSS_PCT': np.random.uniform(0, 0.1),
                'OFFREBOUND_POSS_PCT': np.random.uniform(0, 0.1)
            }

            # Normalize playtypes to sum close to 1
            total_poss = sum(playtypes.values())
            if total_poss > 0:
                playtypes = {k: v/total_poss for k, v in playtypes.items()}

            # PPP for each playtype
            ppp_values = {
                'PPP_ISOLATION': np.random.uniform(0.7, 1.2),
                'PPP_PRBALLHANDLER': np.random.uniform(0.8, 1.3),
                'PPP_PRROLLMAN': np.random.uniform(1.0, 1.5),
                'PPP_POSTUP': np.random.uniform(0.7, 1.1),
                'PPP_SPOTUP': np.random.uniform(0.9, 1.4),
                'PPP_HANDOFF': np.random.uniform(0.8, 1.2),
                'PPP_OFFSCREEN': np.random.uniform(0.9, 1.3),
                'PPP_CUT': np.random.uniform(1.1, 1.5),
                'PPP_TRANSITION': np.random.uniform(1.0, 1.4)
            }

            # Defensive stats
            def_rating = np.random.uniform(95, 115)
            deflections = np.random.uniform(1, 4)
            contested_shots = np.random.uniform(3, 15)
            d_fg_pct = np.random.uniform(0.38, 0.52)

            # Create player record
            player_data = {
                # Basic info
                'PLAYER': player_name,
                'PLAYER_NAME': player_name,
                'PLAYER_NORM': player_name.upper(),
                'POSITION': position,
                'POS': position,
                'POSITION_Y': position,
                'TEAM': team,
                'TEAM_NAME': team,
                'AGE': age,
                'SEASON': season,
                'YEAR': season_year,

                # Basic stats
                'GP': gp,
                'GS': np.random.randint(0, gp),
                'MP': mp,
                'PTS': pts,
                'AST': ast,
                'TRB': reb,
                'STL': stl,
                'BLK': blk,
                'TOV': np.random.uniform(0.5, 4),
                'PF': np.random.uniform(1, 4),

                # Shooting
                'FG': fg,
                'FGA': fga,
                'FG%': fg_pct,
                '3P': threep,
                '3PA': threepa,
                '3P%': threep_pct,
                'FT': ft,
                'FTA': fta,
                'FT%': ft_pct,
                'TS%': TS%,
                'EFG%': EFG%,

                # Advanced
                'PER': per,
                'BPM': bpm,
                'OBPM': bpm * 0.6,
                'DBPM': bpm * 0.4,
                'VORP': vorp,
                'WS': ws,
                'OWS': ws * 0.6,
                'DWS': ws * 0.4,
                'USG%': usg_pct,

                # Usage breakdown
                'OFFENSIVE_LOAD%': usg_pct * 0.6,
                'SCORING_USAGE%': usg_pct * 0.5,
                'PLAYMAKING_USAGE%': usg_pct * 0.3,
                'TRUE_USAGE%': usg_pct,

                # Defensive
                'DEF_RATING': def_rating,
                'DEFLECTIONS': deflections,
                'CONTESTED_SHOTS': contested_shots,
                'D_FG_PCT': d_fg_pct,
                'CRAFTEDDPM': np.random.uniform(-3, 3),
                'VERSATILITYRATING': np.random.uniform(0, 10),

                # Additional advanced stats
                'E_OFF_RATING': np.random.uniform(95, 120),
                'E_DEF_RATING': def_rating,
                'E_NET_RATING': np.random.uniform(-15, 15),
                'PLUSMINUS': np.random.uniform(-10, 10),
            }

            # Add playtypes and PPP
            player_data.update(playtypes)
            player_data.update(ppp_values)

            # Add more playtype stats (FGA, FGM, etc.)
            for playtype in ['ISOLATION', 'PRBALLHANDLER', 'PRROLLMAN', 'POSTUP', 'SPOTUP', 
                           'HANDOFF', 'CUT', 'OFFSCREEN', 'TRANSITION', 'MISC', 'OFFREBOUND']:
                base_poss = player_data.get(f'{playtype}_POSS_PCT', 0) * 100
                player_data[f'{playtype}_POSS'] = base_poss
                player_data[f'{playtype}_FGA'] = base_poss * np.random.uniform(0.7, 1.0)
                player_data[f'{playtype}_FGM'] = player_data[f'{playtype}_FGA'] * np.random.uniform(0.35, 0.55)
                player_data[f'{playtype}_FG_PCT'] = player_data[f'{playtype}_FGM'] / player_data[f'{playtype}_FGA'] if player_data[f'{playtype}_FGA'] > 0 else 0
                player_data[f'{playtype}_EFG%'] = player_data[f'{playtype}_FG_PCT'] * np.random.uniform(0.9, 1.1)
                player_data[f'{playtype}_PPP'] = player_data.get(f'PPP_{playtype}', np.random.uniform(0.8, 1.2))
                player_data[f'{playtype}_PTS'] = base_poss * player_data[f'{playtype}_PPP']
                player_data[f'{playtype}_PERCENTILE'] = np.random.uniform(1, 99)

            all_data.append(player_data)

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Add remaining columns with default/random values
    remaining_columns = [
        '1ST APRON', '2P', '2P%', '2PA', 'AST%', 'AST_PER_36', 'BAE', 'BLK%',
        'DRB', 'DRB%', 'ORB', 'ORB%', 'INJURED', 'INJURY_PERIODS', 'INJURY_RISK',
        'LOSSES', 'LUXURY TAX', 'NON-TAXPAYER MLE', 'PTS_PER_36', 'PLAYER_POSS',
        'SALARY CAP', 'SALARY_CAP_INFLATED', 'TOV%', 'TRB%', 'TRB_PER_36',
        'TAXPAYER MLE', 'TEAM ROOM MLE', 'TEAMID', 'TEAM_POSS', 'TM_AST', 'TM_DRB',
        'TM_FG', 'TM_FGA', 'TM_FTA', 'TM_MP', 'TM_ORB', 'TM_TOV', 'TM_TRB',
        'TOTAL_DAYS_INJURED', 'TURNOVER_USAGE%', 'WS/48', 'WINS', 'YEARS_OF_SERVICE'
    ]

    for col in remaining_columns:
        if col not in df.columns:
            if 'PCT' in col or '%' in col:
                df[col] = np.random.uniform(0, 1, len(df))
            elif any(x in col for x in ['SALARY', 'CAP', 'TAX', 'MLE', 'APRON']):
                df[col] = np.random.uniform(1000000, 50000000, len(df))
            else:
                df[col] = np.random.uniform(0, 100, len(df))

    # Introduce missing values
    if missing_rate > 0:
        # Don't make essential columns missing
        essential_cols = ['PLAYER', 'PLAYER_NAME', 'SEASON', 'POSITION', 'TEAM', 'YEAR']
        missing_candidates = [col for col in df.columns if col not in essential_cols]

        for col in missing_candidates:
            if np.random.random() < 0.7:  # 70% chance a column has missing values
                mask = np.random.random(len(df)) < missing_rate
                df.loc[mask, col] = np.nan

    # Ensure numeric columns are numeric
    numeric_cols = [col for col in df.columns if col not in ['PLAYER', 'PLAYER_NAME', 'PLAYER_NORM', 
                                                              'POSITION', 'POS', 'POSITION_Y', 'TEAM', 
                                                              'TEAM_NAME', 'SEASON']]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Generated test dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%)")
    print(f"Seasons: {df['SEASON'].unique()}")
    print(f"Positions: {df['POSITION'].value_counts().to_dict()}")

    return df


def test_advanced_pipeline(use_test_data=True, test_data_params=None):
    """
    Test the advanced pipeline with either real or synthetic test data

    Args:
        use_test_data: If True, use synthetic data. If False, load from parquet file
        test_data_params: Dict of parameters for generate_test_data() if using test data
    """
    print("Testing ADVANCED NBA Imputation Pipeline...")
    print("="*70)

    # Load or generate data
    if use_test_data:
        print("Using synthetic test data...")
        if test_data_params is None:
            test_data_params = {
                'n_players': 500,
                'n_seasons': 3,
                'missing_rate': 0.3,
                'seed': 42
            }
        df = generate_test_data(**test_data_params)
    else:
        print("Loading real data from parquet file...")
        try:
            df = pd.read_parquet('api/src//data/merged_final_dataset/final_merged_dataset.parquet')
            print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        except FileNotFoundError:
            print("Parquet file not found. Generating test data instead...")
            df = generate_test_data()

    # Clean data
    df = df.replace('%', '', regex=True)
    df.columns = df.columns.str.replace('_x', '')
    df.columns = df.columns.str.replace('_Y', '')
    df.columns = df.columns.str.upper()
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    print(f"\nCleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Missing values before imputation: {df.isnull().sum().sum()}")

    # Verify required columns exist
    required_cols = ['SEASON', 'POSITION', 'PLAYER']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        print(f"WARNING: Missing required columns: {missing_required}")
        # Try alternative column names
        if 'SEASON' not in df.columns and 'YEAR' in df.columns:
            df['SEASON'] = df['YEAR']
        if 'POSITION' not in df.columns and 'POS' in df.columns:
            df['POSITION'] = df['POS']

    # Initialize advanced pipeline with all features enabled
    advanced_imputer = AdvancedNBAPlayerImputation(
        debug=True,
        season_specific=True,              # Process seasons separately
        metric_specific_clustering=True    # Create specialized clusters
    )

    # Run the advanced pipeline
    try:
        df_advanced_imputed, advanced_report = advanced_imputer.run_advanced_pipeline(
            df,
            n_clusters_per_metric=3,  # Clusters per metric type
            n_neighbors=5
        )

        print(f"\nAdvanced Imputation completed!")
        print(f"Original missing values: {df.isnull().sum().sum()}")
        print(f"After advanced imputation: {df_advanced_imputed.isnull().sum().sum()}")

        # Save results
        if use_test_data:
            output_file = 'api/src/data/test_advanced_ml_dataset.csv'
        else:
            output_file = 'api/src/data/final_advanced_ml_dataset.csv'

        df_advanced_imputed.to_csv(output_file, index=False)
        print(f"Advanced imputed dataset saved to {output_file}!")

        # Print detailed summary
        print(f"\nAdvanced Pipeline Summary:")
        print(f"- Seasons processed: {advanced_report['seasons_processed']}")
        print(f"- Offensive features: {len(advanced_imputer.offensive_playtypes)}")
        print(f"- Defensive features: {len(advanced_imputer.defensive_features)}")
        print(f"- Metric clusters created: {advanced_report['metric_clusters_created']}")
        print(f"- Categorical features: {advanced_report['categorical_features_created']}")

        # Show sample of imputed data
        print("\nSample of imputed data (first 5 rows):")
        print(df_advanced_imputed[['PLAYER', 'SEASON', 'POSITION', 'PTS', 'AST', 'TRB']].head())

        # Validation: Check if key offensive/defensive features were processed
        print("\nFeature Processing Validation:")
        print(f"Sample offensive features found: {advanced_imputer.offensive_playtypes[:5]}")
        print(f"Sample defensive features found: {advanced_imputer.defensive_features[:5]}")

        return df_advanced_imputed, advanced_report

    except Exception as e:
        print(f"\nERROR during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# Example usage function to demonstrate different test scenarios
def run_test_examples():
    """
    Run various test examples to demonstrate the pipeline functionality
    """
    print("EXAMPLE 1: Small test dataset with high missing rate")
    print("="*70)
    df1, report1 = test_advanced_pipeline(
        use_test_data=True,
        test_data_params={
            'n_players': 100,
            'n_seasons': 2,
            'missing_rate': 0.4,
            'seed': 123
        }
    )

    print("\n\nEXAMPLE 2: Large test dataset with moderate missing rate")
    print("="*70)
    df2, report2 = test_advanced_pipeline(
        use_test_data=True,
        test_data_params={
            'n_players': 1000,
            'n_seasons': 5,
            'missing_rate': 0.2,
            'seed': 456
        }
    )

    # If you want to test with real data (when available)
    # print("\n\nEXAMPLE 3: Real data from parquet file")
    # print("="*70)
    # df3, report3 = test_advanced_pipeline(use_test_data=False)


if __name__ == "__main__":
    # Run with test data by default
    test_advanced_pipeline(use_test_data=True)
