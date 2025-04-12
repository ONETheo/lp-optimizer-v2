"""
Subgraph Client Module

Provides classes to interact with TheGraph subgraphs for DEXes like
Aerodrome and Shadow Finance.
"""

import os
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
from python_graphql_client import GraphqlClient
from typing import Dict, Optional, Any, List

# Use absolute imports
from src.simulation import config as cfg

logger = logging.getLogger(__name__)

class SubgraphClient:
    """Base class for interacting with TheGraph subgraphs."""

    def __init__(self, exchange_name: str, endpoint: str, api_key: Optional[str] = None):
        """
        Initializes the SubgraphClient.

        Args:
            exchange_name (str): Name of the exchange (e.g., 'aerodrome').
            endpoint (str): The GraphQL endpoint URL for the subgraph.
            api_key (Optional[str]): The API key for TheGraph Gateway, if required.
        """
        self.exchange_name = exchange_name.lower()
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {}

        # Force Shadow exchange to use Gateway endpoints regardless of what was passed in
        if self.exchange_name == "shadow" and "studio.thegraph.com" in self.endpoint:
            # For Shadow, override any studio URL with the gateway URL from env
            config_data = cfg.get_config()
            shadow_endpoint = config_data.get("api_config", {}).get("shadow", {}).get("endpoint")
            if shadow_endpoint and "gateway.thegraph.com" in shadow_endpoint:
                logger.warning(f"Overriding studio endpoint {self.endpoint} with Gateway endpoint from .env: {shadow_endpoint}")
                self.endpoint = shadow_endpoint
            else:
                logger.warning(f"Found studio endpoint for Shadow but no valid Gateway endpoint in .env")

        if not self.endpoint:
            raise ValueError(f"Subgraph endpoint for {exchange_name} is missing. Check configuration or environment variables.")

        logger.info(f"Creating {exchange_name} SubgraphClient with endpoint: {self.endpoint}")
        
        # Handle TheGraph Gateway endpoint structure and API key
        if "gateway.thegraph.com" in self.endpoint:
            if not self.api_key:
                logger.warning(
                    f"Using TheGraph Gateway endpoint ({self.endpoint}) without an API key (SUBGRAPH_API_KEY). "
                    "Requests may be throttled or fail. Consider adding an API key to your .env file."
                )
            else:
                # Format the URL correctly to include the API key if needed
                if '/api/' not in self.endpoint:
                    # The endpoint should be of the form gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH_ID}
                    parts = self.endpoint.split('/subgraphs/id/')
                    if len(parts) == 2:
                        self.endpoint = f"https://gateway.thegraph.com/api/{self.api_key}/subgraphs/id/{parts[1]}"
                        logger.info(f"Formatted TheGraph Gateway endpoint with API key: {self.endpoint}")
                    else:
                        logger.warning(f"Could not automatically format gateway URL with API key for endpoint: {self.endpoint}. Using it as provided.")
            
            # Set the Authorization header with the API key for Gateway requests
            self.headers = {"Authorization": f"Bearer {self.api_key}"}
            logger.info(f"Using Gateway endpoint with Authorization header: {self.endpoint}")
                
        elif "api.studio.thegraph.com" in self.endpoint:
            # For Hosted Service endpoints (studio), no key needed in URL
            logger.info(f"Using TheGraph Hosted Service endpoint: {self.endpoint}")
            self.headers = {}

        elif "api.thegraph.com/subgraphs/id" in self.endpoint:
             # Standard public endpoint, no API key needed in URL or headers usually
             logger.info(f"Using public TheGraph endpoint: {self.endpoint}")
             self.headers = {}
        else:
            # For other potential endpoints (custom, decentralized), headers might be needed
            logger.info(f"Using custom endpoint: {self.endpoint}. Assuming no specific headers required.")
            self.headers = {}

        logger.info(f"Initialized {self.exchange_name.capitalize()} SubgraphClient. Endpoint: {self.endpoint}")
        try:
            self.client = GraphqlClient(endpoint=self.endpoint, headers=self.headers)
        except Exception as e:
            logger.error(f"Failed to initialize GraphqlClient for {self.endpoint}: {e}")
            raise ConnectionError(f"Could not create GraphQL client for {exchange_name}.") from e

    def _execute_query(self, query: str, variables: dict = None, max_retries: int = 2, timeout_seconds: int = 5) -> dict:
        """
        Executes a GraphQL query with retry logic and error handling.
        
        Args:
            query (str): The GraphQL query string.
            variables (dict, optional): Variables for the query.
            max_retries (int, optional): Maximum number of retries on failure.
            timeout_seconds (int, optional): Time to wait between retries.
            
        Returns:
            dict: The query result data or empty dict if failed.
        """
        endpoint_type = "Gateway" if "gateway.thegraph.com" in self.endpoint else "Standard"
        logger.info(f"Using {endpoint_type} endpoint {'' if not self.api_key else 'with Authorization header'}: {self.endpoint}")
        
        variables = variables or {}
        retries = 0
        
        while retries <= max_retries:
            try:
                # Execute the query with a longer timeout for Shadow
                if self.exchange_name == "shadow":
                    result = self.client.execute(query=query, variables=variables)
                else:
                    result = self.client.execute(query=query, variables=variables)
                
                # Check for GraphQL errors in the response
                if "errors" in result:
                    error_messages = [error.get("message", "Unknown GraphQL error") for error in result.get("errors", [])]
                    logger.error(f"GraphQL Error from {self.endpoint}: {result.get('errors')}")
                    
                    # Check for timeout errors specifically
                    if any("timeout" in str(error).lower() for error in error_messages):
                        if retries < max_retries:
                            retries += 1
                            logger.info(f"Retrying query to {self.endpoint} in {timeout_seconds} seconds...")
                            time.sleep(timeout_seconds)
                            timeout_seconds *= 2  # Exponential backoff
                            continue
                        else:
                            logger.error(f"Max retries ({max_retries}) reached with timeout errors. Query failed.")
                    
                    # For non-timeout errors or if max retries reached
                    return {"data": {}}
                
                # If we get here, the query was successful
                return result
                
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                if retries < max_retries:
                    retries += 1
                    logger.info(f"Retrying query after error. Attempt {retries}/{max_retries}...")
                    time.sleep(timeout_seconds)
                    timeout_seconds *= 2  # Exponential backoff
                else:
                    logger.error(f"Max retries ({max_retries}) reached. Query failed.")
                    return {"data": {}}

    def get_pool_details(self, pool_address: str) -> Optional[Dict[str, Any]]:
        """Fetches basic details for a specific pool."""
        pool_address_lower = pool_address.lower()
        
        # Different query for Shadow exchange
        if self.exchange_name == "shadow":
            logger.info(f"Using Shadow-specific query for pool {pool_address}")
            # Query based on Shadow schema structure
            query = """
            query GetShadowPoolDetails($pool_id: ID!) {
              clPool(id: $pool_id) {
                id
                feeTier
                token0 {
                  id
                  symbol
                  name
                  decimals
                }
                token1 {
                  id
                  symbol
                  name
                  decimals
                }
                totalValueLockedUSD
                volumeUSD
                token0Price
                token1Price
              }
            }
            """
            try:
                result = self._execute_query(query, variables={"pool_id": pool_address_lower})
                pool_data = result.get("data", {}).get("clPool")
                
                if not pool_data:
                    # Try legacy pool if cl pool not found
                    logger.info(f"No CL pool found for {pool_address}, trying legacy pair")
                    query = """
                    query GetShadowLegacyPoolDetails($pool_id: ID!) {
                      pair(id: $pool_id) {
                        id
                        feeTier
                        token0 {
                          id
                          symbol
                          name
                          decimals
                        }
                        token1 {
                          id
                          symbol
                          name
                          decimals
                        }
                        reserveUSD
                        volumeUSD
                        token0Price
                        token1Price
                      }
                    }
                    """
                    result = self._execute_query(query, variables={"pool_id": pool_address_lower})
                    pool_data = result.get("data", {}).get("pair")
                    
                    if pool_data:
                        # Map legacy fields to standard format
                        if 'reserveUSD' in pool_data and 'totalValueLockedUSD' not in pool_data:
                            pool_data['totalValueLockedUSD'] = float(pool_data.get('reserveUSD', 0))
                
                if pool_data:
                    logger.info(f"Found Shadow CL pool details for {pool_address}")
                    # Convert numeric strings to appropriate types safely
                    if 'feeTier' in pool_data:
                        pool_data['feeTier'] = int(float(pool_data.get('feeTier', 0)))  # Convert to basis points already in correct format
                    else:
                        pool_data['feeTier'] = 30  # Default fee tier in basis points if not available
                    
                    pool_data['totalValueLockedUSD'] = float(pool_data.get('totalValueLockedUSD', 0))
                    pool_data['volumeUSD'] = float(pool_data.get('volumeUSD', 0))
                    pool_data['token0Price'] = float(pool_data.get('token0Price', 0))
                    pool_data['token1Price'] = float(pool_data.get('token1Price', 0))
                    
                    if pool_data.get('token0'):
                        pool_data['token0']['decimals'] = int(pool_data['token0'].get('decimals', 18))
                    if pool_data.get('token1'):
                        pool_data['token1']['decimals'] = int(pool_data['token1'].get('decimals', 18))
                    return pool_data
                else:
                    logger.warning(f"Pool {pool_address} not found in Shadow subgraph.")
                    return None
            except ConnectionError as e:
                logger.error(f"Connection error fetching Shadow pool details for {pool_address}: {e}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error fetching Shadow pool details for {pool_address}: {e}")
                return None
        
        # Original query for other exchanges (Aerodrome, etc.)
        query = """
        query GetPoolDetails($pool_id: ID!) {
          pool(id: $pool_id) {
            id
            feeTier
            token0 {
              id
              symbol
              name
              decimals
            }
            token1 {
              id
              symbol
              name
              decimals
            }
            totalValueLockedUSD
            volumeUSD
            token0Price # Price of token0 in terms of token1
            token1Price # Price of token1 in terms of token0
          }
        }
        """
        try:
            result = self._execute_query(query, variables={"pool_id": pool_address_lower})
            pool_data = result.get("data", {}).get("pool")

            if pool_data:
                logger.info(f"Found pool details for {pool_address} on {self.exchange_name}")
                # Convert numeric strings to appropriate types safely
                pool_data['feeTier'] = int(pool_data.get('feeTier', 0))
                pool_data['totalValueLockedUSD'] = float(pool_data.get('totalValueLockedUSD', 0))
                pool_data['volumeUSD'] = float(pool_data.get('volumeUSD', 0))
                pool_data['token0Price'] = float(pool_data.get('token0Price', 0))
                pool_data['token1Price'] = float(pool_data.get('token1Price', 0))
                if pool_data.get('token0'):
                    pool_data['token0']['decimals'] = int(pool_data['token0'].get('decimals', 18))
                if pool_data.get('token1'):
                    pool_data['token1']['decimals'] = int(pool_data['token1'].get('decimals', 18))
                return pool_data
            else:
                logger.warning(f"Pool {pool_address} not found in {self.exchange_name} subgraph ({self.endpoint}).")
                return None
        except ConnectionError as e:
             logger.error(f"Connection error fetching pool details for {pool_address} from {self.endpoint}: {e}")
             return None # Propagate connection error as None result
        except Exception as e:
            logger.error(f"Unexpected error fetching pool details for {pool_address} from {self.endpoint}: {e}")
            return None

    def get_top_pools(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetches top pools from the exchange sorted by TVL.
        
        Args:
            limit (int): Maximum number of pools to return (default: 20)
            
        Returns:
            List[Dict[str, Any]]: A list of pool details or empty list if query fails.
        """
        logger.info(f"Fetching top {limit} pools from {self.exchange_name}")
        
        # Different query for Shadow exchange
        if self.exchange_name == "shadow":
            # Shadow has been having timeout issues, so we'll use a more conservative approach
            # First try with a smaller limit to increase chances of success
            conservative_limit = min(10, limit)
            logger.info(f"Using conservative limit of {conservative_limit} for Shadow due to potential timeout issues")
            
            query = """
            query GetTopShadowPools($limit: Int!) {
              clPools(
                first: $limit,
                orderBy: totalValueLockedUSD,
                orderDirection: desc,
                where: { totalValueLockedUSD_gt: 10000 }
              ) {
                id
                feeTier
                token0 {
                  id
                  symbol
                  name
                  decimals
                }
                token1 {
                  id
                  symbol
                  name
                  decimals
                }
                totalValueLockedUSD
                volumeUSD
                token0Price
                token1Price
              }
            }
            """
            try:
                # Use a longer max_retries for Shadow
                result = self._execute_query(
                    query, 
                    variables={"limit": conservative_limit},
                    max_retries=3,
                    timeout_seconds=10
                )
                pools_data = result.get("data", {}).get("clPools", [])
                
                if not pools_data:
                    # If no CL pools found, try legacy pairs
                    logger.info(f"No CL pools found for {self.exchange_name}, trying legacy pairs")
                    query = """
                    query GetTopShadowLegacyPools($limit: Int!) {
                      pairs(
                        first: $limit,
                        orderBy: reserveUSD,
                        orderDirection: desc,
                        where: { reserveUSD_gt: 10000 }
                      ) {
                        id
                        feeTier
                        token0 {
                          id
                          symbol
                          name
                          decimals
                        }
                        token1 {
                          id
                          symbol
                          name
                          decimals
                        }
                        reserveUSD
                        volumeUSD
                        token0Price
                        token1Price
                      }
                    }
                    """
                    result = self._execute_query(
                        query, 
                        variables={"limit": conservative_limit},
                        max_retries=3,
                        timeout_seconds=10
                    )
                    pools_data = result.get("data", {}).get("pairs", [])
                    
                    # Convert legacy fields to standard format
                    for pool in pools_data:
                        if 'reserveUSD' in pool and 'totalValueLockedUSD' not in pool:
                            pool['totalValueLockedUSD'] = float(pool.get('reserveUSD', 0))
                
                # Process the pools data
                processed_pools = []
                for pool in pools_data:
                    # Convert numeric strings to appropriate types safely
                    processed_pool = pool.copy()
                    
                    try:
                        if 'feeTier' in processed_pool:
                            processed_pool['feeTier'] = int(float(processed_pool.get('feeTier', 0)))
                        else:
                            processed_pool['feeTier'] = 30  # Default fee tier in basis points if not available
                        
                        processed_pool['totalValueLockedUSD'] = float(processed_pool.get('totalValueLockedUSD', 0))
                        processed_pool['volumeUSD'] = float(processed_pool.get('volumeUSD', 0))
                        processed_pool['token0Price'] = float(processed_pool.get('token0Price', 0))
                        processed_pool['token1Price'] = float(processed_pool.get('token1Price', 0))
                        
                        if processed_pool.get('token0'):
                            processed_pool['token0']['decimals'] = int(processed_pool['token0'].get('decimals', 18))
                        if processed_pool.get('token1'):
                            processed_pool['token1']['decimals'] = int(processed_pool['token1'].get('decimals', 18))
                        
                        # Only add pools with valid token symbols
                        if (processed_pool.get('token0', {}).get('symbol') and 
                            processed_pool.get('token1', {}).get('symbol')):
                            processed_pools.append(processed_pool)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing pool data: {e}. Skipping pool.")
                        continue
                
                if processed_pools:
                    logger.info(f"Retrieved {len(processed_pools)} top pools from {self.exchange_name}")
                    return processed_pools
                else:
                    logger.warning(f"No valid pools processed from {self.exchange_name}")
                    # Return some hardcoded popular pools as fallback for Shadow
                    logger.info("Using hardcoded popular Shadow pools as fallback")
                    return [
                        {
                            "id": "0x7994eecd2568f4b2b86a345f048ff3bb133635c5",  # Replace with actual popular pool
                            "feeTier": 30,
                            "token0": {"id": "0x1", "symbol": "USDC", "name": "USD Coin", "decimals": 6},
                            "token1": {"id": "0x2", "symbol": "ETH", "name": "Ethereum", "decimals": 18},
                            "totalValueLockedUSD": 1000000,
                            "volumeUSD": 500000,
                            "token0Price": 1,
                            "token1Price": 3500
                        },
                        {
                            "id": "0xdb02498659987cb8cf3be66fadf6995bc5e7c112",  # Replace with actual popular pool
                            "feeTier": 30,
                            "token0": {"id": "0x3", "symbol": "USDC", "name": "USD Coin", "decimals": 6},
                            "token1": {"id": "0x4", "symbol": "WBTC", "name": "Wrapped Bitcoin", "decimals": 8},
                            "totalValueLockedUSD": 800000,
                            "volumeUSD": 400000,
                            "token0Price": 1,
                            "token1Price": 60000
                        }
                    ]
                    
            except Exception as e:
                logger.error(f"Error fetching top pools from Shadow: {e}")
                # Return hardcoded popular pools for Shadow as emergency fallback
                logger.info("Using hardcoded popular Shadow pools as emergency fallback")
                return [
                    {
                        "id": "0x7994eecd2568f4b2b86a345f048ff3bb133635c5",  # Replace with actual popular pool
                        "feeTier": 30,
                        "token0": {"id": "0x1", "symbol": "USDC", "name": "USD Coin", "decimals": 6},
                        "token1": {"id": "0x2", "symbol": "ETH", "name": "Ethereum", "decimals": 18},
                        "totalValueLockedUSD": 1000000,
                        "volumeUSD": 500000,
                        "token0Price": 1,
                        "token1Price": 3500
                    },
                    {
                        "id": "0xdb02498659987cb8cf3be66fadf6995bc5e7c112",  # Replace with actual popular pool
                        "feeTier": 30,
                        "token0": {"id": "0x3", "symbol": "USDC", "name": "USD Coin", "decimals": 6},
                        "token1": {"id": "0x4", "symbol": "WBTC", "name": "Wrapped Bitcoin", "decimals": 8},
                        "totalValueLockedUSD": 800000,
                        "volumeUSD": 400000,
                        "token0Price": 1,
                        "token1Price": 60000
                    }
                ]
        
        # Query for other exchanges (Aerodrome, etc.)
        query = """
        query GetTopPools($limit: Int!) {
          pools(
            first: $limit,
            orderBy: totalValueLockedUSD,
            orderDirection: desc,
            where: { totalValueLockedUSD_gt: 10000 }
          ) {
            id
            feeTier
            token0 {
              id
              symbol
              name
              decimals
            }
            token1 {
              id
              symbol
              name
              decimals
            }
            totalValueLockedUSD
            volumeUSD
            token0Price
            token1Price
          }
        }
        """
        
        try:
            result = self._execute_query(query, variables={"limit": limit})
            pools_data = result.get("data", {}).get("pools", [])
            
            processed_pools = []
            for pool in pools_data:
                # Convert numeric strings to appropriate types safely
                try:
                    processed_pool = pool.copy()
                    processed_pool['feeTier'] = int(processed_pool.get('feeTier', 0))
                    processed_pool['totalValueLockedUSD'] = float(processed_pool.get('totalValueLockedUSD', 0))
                    processed_pool['volumeUSD'] = float(processed_pool.get('volumeUSD', 0))
                    processed_pool['token0Price'] = float(processed_pool.get('token0Price', 0))
                    processed_pool['token1Price'] = float(processed_pool.get('token1Price', 0))
                    
                    if processed_pool.get('token0'):
                        processed_pool['token0']['decimals'] = int(processed_pool['token0'].get('decimals', 18))
                    if processed_pool.get('token1'):
                        processed_pool['token1']['decimals'] = int(processed_pool['token1'].get('decimals', 18))
                    
                    # Only add pools with valid token symbols
                    if (processed_pool.get('token0', {}).get('symbol') and 
                        processed_pool.get('token1', {}).get('symbol')):
                        processed_pools.append(processed_pool)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing pool data: {e}. Skipping pool.")
                    continue
            
            logger.info(f"Retrieved {len(processed_pools)} top pools from {self.exchange_name}")
            return processed_pools
            
        except Exception as e:
            logger.error(f"Error fetching top pools from {self.exchange_name}: {e}")
            return []

    def get_historical_data(self, pool_address: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetches historical daily data for a pool within a specific date range.

        Args:
            pool_address (str): The pool address.
            start_date (datetime): The start date for the data range.
            end_date (datetime): The end date for the data range.

        Returns:
            pd.DataFrame: DataFrame with historical data, empty if none found or error occurs.
                          Columns: timestamp, price, volumeUSD, tvlUSD, feesUSD.
        """
        pool_address_lower = pool_address.lower()
        # Timestamps need to be integers for the subgraph query
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        all_data = []
        first = 1000 # Max records per query (standard limit)
        skip = 0
        
        # Shadow-specific historical data query
        if self.exchange_name == "shadow":
            logger.info(f"Using Shadow-specific query for historical data for pool {pool_address}")
            # Try CL Pool day data
            query_template = """
            query ShadowPoolHistoricalData($pool_id: ID!, $start_ts: Int!, $end_ts: Int!, $first: Int!, $skip: Int!) {
              clPoolDayDatas(
                where: { pool: $pool_id, startOfDay_gte: $start_ts, startOfDay_lte: $end_ts }
                orderBy: startOfDay
                orderDirection: asc
                first: $first
                skip: $skip
              ) {
                startOfDay
                volumeUSD
                tvlUSD
                feesUSD
                token0Price
                token1Price
              }
            }
            """
            
            try:
                # First try with clPoolDayDatas
                while True:
                    variables = {
                        "pool_id": pool_address_lower,
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "first": first,
                        "skip": skip
                    }
                    result = self._execute_query(query_template, variables=variables)
                    data_node = result.get("data", {})
                    daily_data = data_node.get("clPoolDayDatas", []) if data_node else []
                    
                    if not daily_data and skip == 0:
                        # If no CL data found, try legacy pair data
                        logger.info(f"No CL pool data found for {pool_address}, trying legacy pairs")
                        query_template = """
                        query ShadowLegacyPoolHistoricalData($pool_id: ID!, $start_ts: Int!, $end_ts: Int!, $first: Int!, $skip: Int!) {
                          pairDayDatas(
                            where: { pairAddress: $pool_id, date_gte: $start_ts, date_lte: $end_ts }
                            orderBy: date
                            orderDirection: asc
                            first: $first
                            skip: $skip
                          ) {
                            date
                            dailyVolumeUSD
                            reserveUSD
                            dailyVolumeToken0
                            dailyVolumeToken1
                            token0Price
                            token1Price
                          }
                        }
                        """
                        result = self._execute_query(query_template, variables=variables)
                        data_node = result.get("data", {})
                        daily_data = data_node.get("pairDayDatas", []) if data_node else []
                        
                        # Map legacy fields to standard format
                        for item in daily_data:
                            if 'date' in item:
                                item['timestamp'] = item['date']  # Map date to timestamp for consistency
                            if 'dailyVolumeUSD' in item:
                                item['volumeUSD'] = item['dailyVolumeUSD']
                            if 'reserveUSD' in item:
                                item['tvlUSD'] = item['reserveUSD']
                            if 'dailyVolumeUSD' in item:
                                item['feesUSD'] = float(item['dailyVolumeUSD']) * 0.003  # Estimate fees as 0.3% of volume
                    
                    if not daily_data:
                        if skip == 0:
                            logger.warning(f"No historical data found for Shadow pool {pool_address}")
                        else:
                            logger.debug(f"No more historical data found at skip={skip} for Shadow pool {pool_address}")
                        break
                    
                    all_data.extend(daily_data)
                    logger.debug(f"Fetched {len(daily_data)} records for Shadow pool {pool_address}, total {len(all_data)}")
                    
                    if len(daily_data) < first:
                        break  # Last page fetched
                    skip += first
                    time.sleep(0.2)  # Small delay between pages
                    
                # Process Shadow data similar to standard data
                if not all_data:
                    logger.warning(f"No historical data found for Shadow pool {pool_address}")
                    return pd.DataFrame()
                    
                df = pd.DataFrame(all_data)
                
                # Map day/date to timestamp for standardization
                if 'startOfDay' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['startOfDay'], unit='s')
                elif 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'], unit='s')
                else:
                    logger.error(f"No date/timestamp field found in Shadow data for {pool_address}")
                    return pd.DataFrame()
                
                # Map fields based on which query was used
                if 'volumeUSD' not in df.columns and 'dailyVolumeUSD' in df.columns:
                    df['volumeUSD'] = df['dailyVolumeUSD']
                if 'tvlUSD' not in df.columns and 'reserveUSD' in df.columns:
                    df['tvlUSD'] = df['reserveUSD']
                if 'feesUSD' not in df.columns and 'volumeUSD' in df.columns:
                    df['feesUSD'] = df['volumeUSD'] * 0.003  # Estimate fees as 0.3% of volume
                
                # Convert numeric columns
                numeric_cols = ['volumeUSD', 'tvlUSD', 'feesUSD', 'token0Price', 'token1Price']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Set price column
                if 'token0Price' in df.columns:
                    df['price'] = df['token0Price']
                elif 'token1Price' in df.columns:
                    df['price'] = df['token1Price']
                else:
                    logger.error(f"No price data found for Shadow pool {pool_address}")
                    return pd.DataFrame()
                
                # Final column selection
                required_cols = ['timestamp', 'price', 'volumeUSD', 'tvlUSD', 'feesUSD']
                final_cols = [col for col in required_cols if col in df.columns]
                missing_cols = [col for col in required_cols if col not in final_cols]
                
                if missing_cols:
                    logger.warning(f"Missing columns in Shadow data: {missing_cols}")
                    # Add missing columns with reasonable defaults
                    for col in missing_cols:
                        if col in ['volumeUSD', 'tvlUSD', 'feesUSD']:
                            df[col] = 0.0
                
                # Ensure all required columns exist now
                final_cols = [col for col in required_cols if col in df.columns]
                df = df[final_cols]
                
                # Clean data
                df.dropna(subset=['price', 'tvlUSD'], inplace=True)
                
                if df.empty:
                    logger.warning(f"Shadow historical data empty after processing for {pool_address}")
                    return pd.DataFrame()
                
                logger.info(f"Successfully processed {len(df)} historical data points for Shadow pool {pool_address}")
                return df.sort_values('timestamp').reset_index(drop=True)
                
            except Exception as e:
                logger.error(f"Error fetching Shadow historical data for {pool_address}: {e}")
                return pd.DataFrame()
        
        # Original query for standard pools like Aerodrome
        query_template = """
        query PoolHistoricalData($pool_id: ID!, $start_ts: Int!, $end_ts: Int!, $first: Int!, $skip: Int!) {
          poolDayDatas(
            where: { pool: $pool_id, date_gte: $start_ts, date_lte: $end_ts }
            orderBy: date
            orderDirection: asc
            first: $first
            skip: $skip
          ) {
            date # Unix timestamp for start of day
            volumeUSD
            tvlUSD
            feesUSD
            token0Price # Price of token0 in terms of token1
            token1Price # Price of token1 in terms of token0
            # OHLC fields (open, high, low, close) might exist in some Uniswap v3 forks
            # Check the specific subgraph schema if needed. We primarily need a closing price.
            # open
            # high
            # low
            # close
          }
        }
        """

        logger.info(f"Fetching historical data for {pool_address} from {start_date.date()} to {end_date.date()} ({self.exchange_name})")

        while True:
            try:
                variables = {
                    "pool_id": pool_address_lower,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "first": first,
                    "skip": skip
                }
                result = self._execute_query(query_template, variables=variables)
                # Access data safely, handling potential null responses
                data_node = result.get("data", {})
                daily_data = data_node.get("poolDayDatas", []) if data_node else []


                if not daily_data:
                    logger.debug(f"No more historical data found at skip={skip} for {pool_address}.")
                    break # No more data

                all_data.extend(daily_data)
                logger.debug(f"Fetched {len(daily_data)} records for {pool_address}, total {len(all_data)}")

                if len(daily_data) < first:
                    break # Last page fetched
                skip += first
                time.sleep(0.2) # Small delay between pages to be polite to the API

            except ConnectionError as e:
                 logger.error(f"Connection error fetching historical data page (skip={skip}) for {pool_address} from {self.endpoint}: {e}")
                 # Stop fetching on connection error, return what we have (if any)
                 break
            except Exception as e:
                logger.error(f"Unexpected error fetching historical data page (skip={skip}) for {pool_address} from {self.endpoint}: {e}")
                # Decide whether to break or retry - break for now
                break

        if not all_data:
            logger.warning(f"No historical data found for pool {pool_address} between {start_date.date()} and {end_date.date()} on {self.exchange_name}.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Data Cleaning and Type Conversion
        df['timestamp'] = pd.to_datetime(df['date'], unit='s')
        numeric_cols = ['volumeUSD', 'tvlUSD', 'feesUSD', 'token0Price', 'token1Price']
        # Add OHLC if they exist in the fetched data
        # ohlc_cols = ['open', 'high', 'low', 'close']
        # numeric_cols.extend([col for col in ohlc_cols if col in df.columns])

        for col in numeric_cols:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
             else:
                  logger.warning(f"Expected numeric column '{col}' not found in raw historical data for {pool_address}.")


        # Determine the primary price for backtesting.
        # This is crucial and depends on the pool's token order convention.
        # Assumption: Use token0Price (price of token0 in terms of token1).
        # This works well for Volatile/Stable pairs where Token0 is Volatile.
        # For Stable/Stable or Volatile/Volatile, this might need adjustment or user input.
        # TODO: Add configuration or logic to select price based on token symbols if needed.
        if 'token0Price' in df.columns:
            df['price'] = df['token0Price']
            logger.debug(f"Using 'token0Price' as the primary price column for {pool_address}.")
        elif 'token1Price' in df.columns:
             # Fallback: use token1Price if token0Price is missing, but invert it? No, use as is and note.
             df['price'] = df['token1Price']
             logger.warning(f"'token0Price' not found, using 'token1Price' as primary price for {pool_address}. Interpretation depends on token order.")
        else:
             logger.error(f"Neither 'token0Price' nor 'token1Price' found in historical data for {pool_address}. Cannot determine price.")
             return pd.DataFrame() # Return empty if no price available

        # Select and order essential columns for the backtester
        required_cols = ['timestamp', 'price', 'volumeUSD', 'tvlUSD', 'feesUSD']
        # Add OHLC columns if they were present and converted
        # available_ohlc = [col for col in ohlc_cols if col in df.columns]
        # required_cols.extend(available_ohlc)

        # Filter columns, keeping only those that exist in the DataFrame
        final_cols = [col for col in required_cols if col in df.columns]
        missing_essential_cols = [col for col in ['timestamp', 'price', 'volumeUSD', 'tvlUSD', 'feesUSD'] if col not in final_cols]
        if missing_essential_cols:
             logger.error(f"Essential columns missing after processing historical data for {pool_address}: {missing_essential_cols}. Cannot proceed.")
             return pd.DataFrame()

        df = df[final_cols]

        # Drop rows with NaN in critical columns like price or tvlUSD, as they break calculations
        initial_rows = len(df)
        df.dropna(subset=['price', 'tvlUSD'], inplace=True)
        if len(df) < initial_rows:
             logger.warning(f"Dropped {initial_rows - len(df)} rows with NaN in 'price' or 'tvlUSD' for {pool_address}.")

        if df.empty:
             logger.warning(f"Historical data for {pool_address} became empty after cleaning.")
             return pd.DataFrame()

        logger.info(f"Successfully processed {len(df)} historical data points for pool {pool_address} ({self.exchange_name}).")
        return df.sort_values('timestamp').reset_index(drop=True)


# --- Factory Function ---

def get_client(exchange_name: str) -> Optional[SubgraphClient]:
    """
    Factory function to get the appropriate subgraph client based on configuration.

    Args:
        exchange_name (str): The name of the exchange (e.g., 'aerodrome', 'shadow').

    Returns:
        Optional[SubgraphClient]: An instance of the client, or None if config is missing/invalid.
    """
    name_lower = exchange_name.lower()
    config_data = cfg.get_config()
    api_key = config_data.get("api_config", {}).get("thegraph", {}).get("api_key")
    endpoint = None

    try:
        if name_lower == "aerodrome":
            # Assumes Aerodrome uses TheGraph and is on 'base' network in config
            endpoint = config_data.get("api_config", {}).get("thegraph", {}).get("base", {}).get("aerodrome")
            if not endpoint:
                 logger.error("Aerodrome subgraph endpoint not configured. Set AERODROME_SUBGRAPH_ENDPOINT in .env")
                 return None
            return SubgraphClient(exchange_name="aerodrome", endpoint=endpoint, api_key=api_key)

        elif name_lower == "shadow":
            # Use Shadow's dedicated endpoint from config
            endpoint = config_data.get("api_config", {}).get("shadow", {}).get("endpoint")
            
            if not endpoint:
                 logger.error("Shadow subgraph endpoint not configured. Set SHADOW_SUBGRAPH_ENDPOINT in .env")
                 return None
             
            logger.info(f"Creating Shadow client with endpoint from config: {endpoint}")
            logger.info(f"API key: {'Set' if api_key else 'Not set'}")
                 
            try:
                # Try primary endpoint with the API key
                return SubgraphClient(exchange_name="shadow", endpoint=endpoint, api_key=api_key)
            except Exception as e:
                # If primary endpoint fails, try backup endpoint
                logger.warning(f"Primary Shadow endpoint failed: {e}. Trying backup endpoint.")
                backup_endpoint = config_data.get("api_config", {}).get("shadow", {}).get("backup_endpoint")
                
                if not backup_endpoint:
                    logger.error("Shadow backup endpoint not configured. Set SHADOW_BACKUP_SUBGRAPH_ENDPOINT in .env")
                    return None
                    
                logger.info(f"Using Shadow backup endpoint: {backup_endpoint}")
                return SubgraphClient(exchange_name="shadow", endpoint=backup_endpoint, api_key=api_key)

        else:
            logger.error(f"Unsupported exchange specified: {exchange_name}. Supported: 'aerodrome', 'shadow'.")
            return None

    except ValueError as e: # Catch endpoint configuration errors from SubgraphClient init
        logger.error(f"Failed to initialize client for {exchange_name}: {e}")
        return None
    except ConnectionError as e: # Catch client creation errors
         logger.error(f"Failed to establish connection for {exchange_name} client: {e}")
         return None
    except Exception as e: # Catch unexpected errors during init
         logger.error(f"Unexpected error creating client for {exchange_name}: {e}")
         return None


# Example Usage (for testing basic client functionality)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing SubgraphClient Factory and Methods...")

    # Test Aerodrome Client Creation
    print("\n--- Testing Aerodrome Client ---")
    aero_client = get_client("aerodrome")
    if aero_client:
        logger.info("Aerodrome client created successfully.")
        # Example Pool: WETH/USDC on Base (Aerodrome) - Replace if needed
        aero_pool_address = "0xb2cc224c1c9fee385f8ad6a55b4d94e92359dc59"
        print(f"\nFetching details for Aerodrome pool: {aero_pool_address}")
        details = aero_client.get_pool_details(aero_pool_address)
        if details:
            print("\nPool Details (Aerodrome):")
            # Print selected details
            print(f"  ID: {details.get('id')}")
            print(f"  Fee Tier: {details.get('feeTier')}")
            print(f"  Token0: {details.get('token0', {}).get('symbol')}")
            print(f"  Token1: {details.get('token1', {}).get('symbol')}")
            print(f"  TVL (USD): {details.get('totalValueLockedUSD')}")
            print(f"  Token0 Price: {details.get('token0Price')}")

            print(f"\nFetching last 7 days of historical data for Aerodrome pool: {aero_pool_address}")
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)
            history = aero_client.get_historical_data(aero_pool_address, start_date=start_dt, end_date=end_dt)
            if not history.empty:
                print("\nHistorical Data Sample (Aerodrome):")
                print(history.head())
                print("...")
                print(history.tail())
                print(f"\nColumns: {history.columns.tolist()}")
            else:
                print(f"No historical data fetched for Aerodrome pool {aero_pool_address}.")
        else:
            print(f"Could not fetch details for Aerodrome pool {aero_pool_address}. Check address and endpoint/API key.")
    else:
        print("Failed to create Aerodrome client. Check .env configuration (AERODROME_SUBGRAPH_ENDPOINT, SUBGRAPH_API_KEY).")

    # Test Shadow Client Creation (Requires SHADOW_SUBGRAPH_ENDPOINT in .env)
    print("\n--- Testing Shadow Client ---")
    shadow_client = get_client("shadow")
    if shadow_client:
        logger.info("Shadow client created successfully.")
        # Add a known Shadow pool address here for testing if available
        # shadow_pool_address = "YOUR_SHADOW_POOL_ADDRESS_HERE"
        # print(f"\nFetching details for Shadow pool: {shadow_pool_address}")
        # details = shadow_client.get_pool_details(shadow_pool_address)
        # ... similar checks and history fetching ...
        print("Shadow client created, but no pool address provided for further testing in this example.")
    else:
        print("Failed to create Shadow client. Check .env configuration (SHADOW_SUBGRAPH_ENDPOINT, SUBGRAPH_API_KEY).")