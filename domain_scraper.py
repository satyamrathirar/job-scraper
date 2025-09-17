#!/usr/bin/env python3
"""
Comprehensive Website Domain Scraper

This script reads company names and descriptions from an Excel file,
searches for their websites using multiple strategies, extracts domains,
and saves the results to a file in the same order.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import tldextract
from urllib.parse import urljoin, urlparse
import csv
import os
from typing import List, Optional, Tuple
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('domain_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveWebsiteScraper:
    def __init__(self, delay: float = 1.5):
        """
        Initialize the website scraper.
        
        Args:
            delay: Delay between requests in seconds to be respectful to servers
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.session.timeout = 10
    
    def extract_domain(self, url: str) -> Optional[str]:
        """Extract clean domain from URL."""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Use tldextract for better domain parsing
            extracted = tldextract.extract(domain)
            if extracted.domain and extracted.suffix:
                clean_domain = f"{extracted.domain}.{extracted.suffix}"
                return clean_domain
            
            return None
            
        except Exception:
            return None
    
    def search_with_google(self, company_name: str, description: str = "") -> Optional[str]:
        """Search using Google search."""
        try:
            # Create comprehensive search query with description
            if description and len(description.strip()) > 0:
                # Use key terms from description
                desc_words = description.lower().split()
                key_desc_words = [word for word in desc_words if len(word) > 4 and word not in ['company', 'business', 'solutions', 'services']][:3]
                search_query = f'"{company_name}" {" ".join(key_desc_words)} official website site:'
            else:
                search_query = f'"{company_name}" official website'
            
            logger.info(f"Google search for: {company_name}")
            
            # Google search URL
            search_url = "https://www.google.com/search"
            params = {
                'q': search_query,
                'num': 10,
                'hl': 'en'
            }
            
            response = self.session.get(search_url, params=params)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result links
            result_links = []
            
            # Try different selectors for Google results
            selectors = [
                'div.g a[href]',
                'h3 a[href]',
                '.yuRUbf a[href]',
                'a[href*="http"]'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href', '')
                    if href and not any(x in href for x in ['google.com', 'youtube.com', 'translate.google']):
                        result_links.append(href)
                
                if result_links:
                    break
            
            # Extract and validate domains
            for url in result_links[:10]:
                domain = self.extract_domain(url)
                if domain and self.is_valid_company_domain(domain, company_name, description):
                    logger.info(f"Found domain via Google search: {domain}")
                    return domain
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in Google search: {str(e)}")
            return None
    
    def search_with_bing(self, company_name: str, description: str = "") -> Optional[str]:
        """Search using Bing with improved query using description."""
        try:
            # Create comprehensive search query with description
            if description and len(description.strip()) > 0:
                # Use key words from description for better search
                desc_words = description.split()[:10]  # First 10 words
                search_query = f'"{company_name}" official website {" ".join(desc_words)}'
            else:
                search_query = f'"{company_name}" official website'
            
            logger.info(f"Bing search query: {search_query}")
            
            # Bing search URL
            search_url = "https://www.bing.com/search"
            params = {
                'q': search_query,
                'count': 20,  # Get more results
                'first': 1
            }
            
            response = self.session.get(search_url, params=params)
            if response.status_code != 200:
                logger.warning(f"Bing search failed with status: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result links with multiple selectors
            result_links = []
            
            # Try different selectors for Bing results
            selectors = [
                'h2 a[href]',
                '.b_title a[href]',
                '.b_algo h2 a[href]',
                'cite',
                '.b_attribution cite'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    if selector == 'cite' or 'attribution' in selector:
                        # For cite elements, get the text content as URL
                        url = element.get_text().strip()
                        if url and not url.startswith('http'):
                            url = 'https://' + url
                    else:
                        # For anchor elements, get href
                        url = element.get('href', '')
                    
                    if url and self.is_valid_url(url):
                        result_links.append(url)
                
                if result_links:
                    break
            
            # Process and validate domains
            seen_domains = set()
            for url in result_links[:15]:  # Check more results
                domain = self.extract_domain(url)
                if domain and domain not in seen_domains and self.is_valid_company_domain(domain, company_name):
                    # Additional validation: check if the URL actually works
                    if self.verify_domain_works(domain):
                        logger.info(f"Found and verified domain via Bing: {domain}")
                        return domain
                    seen_domains.add(domain)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in Bing search: {str(e)}")
            return None
    
    def search_with_duckduckgo(self, company_name: str, description: str = "") -> Optional[str]:
        """Search using DuckDuckGo with improved description-based queries."""
        try:
            # Create comprehensive search query with description
            if description and len(description.strip()) > 0:
                desc_words = description.split()[:8]  # First 8 words from description
                search_query = f'"{company_name}" official website {" ".join(desc_words)}'
            else:
                search_query = f'"{company_name}" official website'
            
            logger.info(f"DuckDuckGo search query: {search_query}")
            
            search_url = "https://duckduckgo.com/html/"
            params = {
                'q': search_query,
                'kl': 'us-en',
                's': '0'  # Start from first result
            }
            
            response = self.session.get(search_url, params=params)
            if response.status_code != 200:
                logger.warning(f"DuckDuckGo search failed with status: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract search result links
            result_links = []
            
            # Try multiple selectors for DuckDuckGo
            selectors = [
                'a.result__a[href]',
                '.result__title a[href]',
                '.web-result__title a[href]',
                'h2.result__title a[href]'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href', '')
                    if href:
                        # Handle DuckDuckGo redirect URLs
                        if 'duckduckgo.com/l/' in href:
                            try:
                                # Extract actual URL from DuckDuckGo redirect
                                import urllib.parse
                                if 'uddg=' in href:
                                    encoded_url = href.split('uddg=')[1].split('&')[0]
                                    actual_url = urllib.parse.unquote(encoded_url)
                                    result_links.append(actual_url)
                            except:
                                continue
                        else:
                            result_links.append(href)
                
                if result_links:
                    break
            
            # Also try to find domains in result snippets
            snippets = soup.find_all('a', class_='result__snippet')
            for snippet in snippets:
                text = snippet.get_text()
                # Look for domain patterns in text
                import re
                domain_pattern = r'\b([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
                domains_in_text = re.findall(domain_pattern, text)
                for domain_match in domains_in_text:
                    if isinstance(domain_match, tuple):
                        domain = domain_match[0] + domain_match[1]
                    else:
                        domain = domain_match
                    if self.is_valid_company_domain(domain, company_name):
                        result_links.append(f"https://{domain}")
            
            # Process and validate domains
            seen_domains = set()
            for url in result_links[:15]:
                domain = self.extract_domain(url)
                if domain and domain not in seen_domains and self.is_valid_company_domain(domain, company_name):
                    if self.verify_domain_works(domain):
                        logger.info(f"Found and verified domain via DuckDuckGo: {domain}")
                        return domain
                    seen_domains.add(domain)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in DuckDuckGo search: {str(e)}")
            return None
    
    def search_with_google(self, company_name: str, description: str = "") -> Optional[str]:
        """Search using Google as a last resort."""
        try:
            # Create search query with description
            if description and len(description.strip()) > 0:
                desc_words = description.split()[:6]  # First 6 words from description
                search_query = f'"{company_name}" official website {" ".join(desc_words)}'
            else:
                search_query = f'"{company_name}" official website'
            
            logger.info(f"Google search query: {search_query}")
            
            search_url = "https://www.google.com/search"
            params = {
                'q': search_query,
                'num': 20,  # Get more results
                'hl': 'en'
            }
            
            response = self.session.get(search_url, params=params)
            if response.status_code != 200:
                logger.warning(f"Google search failed with status: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search result links
            result_links = []
            
            # Try different selectors for Google results
            selectors = [
                'div.g h3 a[href]',
                'div.g a[href]',
                'h3 a[href]',
                'cite',
                '.iUh30'  # Google's cite element
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    if selector == 'cite' or 'iUh30' in selector:
                        # For cite elements, get the text content as URL
                        url = element.get_text().strip()
                        if url and not url.startswith('http'):
                            url = 'https://' + url
                    else:
                        # For anchor elements, get href
                        url = element.get('href', '')
                        # Clean Google redirect URLs
                        if 'google.com/url?' in url:
                            try:
                                import urllib.parse
                                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                                if 'url' in parsed:
                                    url = parsed['url'][0]
                            except:
                                continue
                    
                    if url and self.is_valid_url(url):
                        result_links.append(url)
                
                if result_links:
                    break
            
            # Process and validate domains
            seen_domains = set()
            for url in result_links[:15]:
                domain = self.extract_domain(url)
                if domain and domain not in seen_domains and self.is_valid_company_domain(domain, company_name):
                    if self.verify_domain_works(domain):
                        logger.info(f"Found and verified domain via Google: {domain}")
                        return domain
                    seen_domains.add(domain)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in Google search: {str(e)}")
            return None
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not from excluded domains."""
        if not url:
            return False
        
        excluded_patterns = [
            'google.com', 'bing.com', 'duckduckgo.com', 'yahoo.com',
            'facebook.com', 'twitter.com', 'linkedin.com', 'youtube.com',
            'instagram.com', 'wikipedia.org', 'bloomberg.com', 'forbes.com'
        ]
        
        for pattern in excluded_patterns:
            if pattern in url.lower():
                return False
        
        return True
    
    def verify_domain_works(self, domain: str) -> bool:
        """Verify that a domain actually works by making a HEAD request."""
        try:
            response = self.session.head(f"https://{domain}", timeout=5, allow_redirects=True)
            return response.status_code == 200
        except:
            try:
                response = self.session.head(f"http://{domain}", timeout=5, allow_redirects=True)
                return response.status_code == 200
            except:
                return False
    def is_valid_company_domain(self, domain: str, company_name: str) -> bool:
        """Check if domain is likely to belong to the company with improved matching."""
        if not domain:
            return False
        
        # Skip common non-company domains
        excluded_domains = {
            'google.com', 'facebook.com', 'twitter.com', 'linkedin.com',
            'youtube.com', 'instagram.com', 'wikipedia.org', 'bloomberg.com',
            'forbes.com', 'reuters.com', 'yahoo.com', 'bing.com',
            'duckduckgo.com', 'crunchbase.com', 'glassdoor.com',
            'indeed.com', 'monster.com', 'careerbuilder.com', 'github.com',
            'stackoverflow.com', 'reddit.com', 'amazon.com', 'apple.com'
        }
        
        if domain.lower() in excluded_domains:
            return False
        
        # Get clean company name words (remove common words and punctuation)
        company_clean = re.sub(r'[^\w\s]', ' ', company_name.lower())
        company_words = [w for w in company_clean.split() if len(w) > 2 and w not in {'inc', 'ltd', 'llc', 'corp', 'corporation', 'company', 'co', 'the', 'and', 'for', 'of', 'group'}]
        
        domain_name = domain.split('.')[0].lower()
        domain_parts = re.split(r'[-_]', domain_name)
        
        # Score the domain based on different matching criteria
        score = 0
        
        # Exact word matches
        for word in company_words:
            if word in domain_name:
                score += 3
            for part in domain_parts:
                if word == part:
                    score += 5
                elif word in part and len(word) > 3:
                    score += 2
        
        # Acronym matching (for names like "Forest Stewardship Council" -> "fsc")
        if len(company_words) > 1:
            acronym = ''.join([w[0] for w in company_words])
            if acronym == domain_name or acronym in domain_parts:
                score += 8
        
        # Partial matching for compound words
        for word in company_words:
            if len(word) > 4:
                for part in domain_parts:
                    if len(part) > 4 and (word[:4] in part or part[:4] in word):
                        score += 1
        
        # Domain length penalty (very long domains are usually not official)
        if len(domain_name) > 20:
            score -= 2
        
        # Common domain patterns boost
        if domain.endswith(('.com', '.org', '.net')):
            score += 1
        
        # Return True if score is high enough
        return score >= 3
    
    def find_company_website(self, company_name: str, description: str = "") -> Optional[str]:
        """
        Find company website using web search only (no pattern matching).
        
        Args:
            company_name: Name of the company
            description: Company description for better search results
            
        Returns:
            Domain of the company's website or None if not found
        """
        logger.info(f"Searching for website: {company_name}")
        
        # Strategy 1: Search with Bing using description
        domain = self.search_with_bing(company_name, description)
        if domain:
            return domain
        
        # Strategy 2: Search with DuckDuckGo using description
        time.sleep(0.5)
        domain = self.search_with_duckduckgo(company_name, description)
        if domain:
            return domain
        
        # Strategy 3: Try Google search as fallback
        time.sleep(0.5)
        domain = self.search_with_google(company_name, description)
        if domain:
            return domain
        
        # Strategy 4: Try searches with variations of the company name
        variations = self.get_company_name_variations(company_name)
        for variation in variations:
            time.sleep(0.5)
            domain = self.search_with_bing(variation, description)
            if domain:
                return domain
                
            time.sleep(0.5)
            domain = self.search_with_duckduckgo(variation, description)
            if domain:
                return domain
        
        # Strategy 5: As last resort, try pattern matching with common TLDs
        domain = self.try_limited_pattern_matching(company_name)
        if domain:
            return domain
        
        logger.warning(f"Could not find website for: {company_name}")
        return None
    
    def get_company_name_variations(self, company_name: str) -> List[str]:
        """Generate variations of company name for search."""
        variations = []
        
        # Remove common business suffixes
        clean_name = re.sub(r'\b(inc|ltd|llc|corp|corporation|company|co|group|initiative|council|energy|solutions?|technologies?|tech|systems?)\b', '', company_name.lower()).strip()
        if clean_name != company_name.lower():
            variations.append(clean_name.title())
        
        # Try just the first word if it's unique enough
        words = company_name.split()
        if len(words) > 1 and len(words[0]) > 3:
            variations.append(words[0])
        
        # Try first two words
        if len(words) > 2:
            variations.append(' '.join(words[:2]))
        
        return variations[:3]  # Limit to 3 variations
    
    def try_limited_pattern_matching(self, company_name: str) -> Optional[str]:
        """Try limited pattern matching with .net, .org domains only."""
        try:
            clean_name = re.sub(r'[^\w\s]', '', company_name.lower())
            words = clean_name.split()
            
            if words:
                main_word = words[0]
                # Only try .net and .org for edge cases
                patterns = [f"{main_word}.net", f"{main_word}.org"]
                
                # Test each pattern
                for domain in patterns:
                    try:
                        response = self.session.head(
                            f"https://{domain}",
                            timeout=3,
                            allow_redirects=True
                        )
                        if response.status_code == 200:
                            logger.info(f"Found working domain via limited pattern: {domain}")
                            return domain
                    except:
                        continue
            
            return None
            
        except Exception:
            return None
    
    def process_excel_file(self, excel_path: str, output_path: str = 'scraped_domains.csv'):
        """
        Process the Excel file and scrape domains for all companies.
        
        Args:
            excel_path: Path to the Excel file
            output_path: Path to save the results
        """
        try:
            # Read Excel file
            logger.info(f"Reading Excel file: {excel_path}")
            df = pd.read_excel(excel_path)
            
            # Check if required columns exist
            if 'Company Name' not in df.columns:
                raise ValueError("'Company Name' column not found in Excel file")
            
            # Prepare results list
            results = []
            total_companies = len(df)
            found_count = 0
            
            # Process each company
            for index, row in df.iterrows():
                company_name = str(row['Company Name']).strip()
                description = str(row.get('Company Description', '')).strip() if 'Company Description' in df.columns else ''
                
                if company_name and company_name.lower() != 'nan':
                    logger.info(f"Processing {index + 1}/{total_companies}: {company_name}")
                    
                    # Search for website domain
                    domain = self.find_company_website(company_name, description)
                    
                    if domain:
                        found_count += 1
                    
                    results.append({
                        'Row_Number': index + 1,
                        'Company_Name': company_name,
                        'Company_Description': description[:100] + '...' if len(description) > 100 else description,
                        'Found_Domain': domain if domain else 'Not Found',
                        'Search_Status': 'Success' if domain else 'Failed'
                    })
                    
                    # Add delay between requests
                    if index < total_companies - 1:
                        time.sleep(self.delay)
                else:
                    results.append({
                        'Row_Number': index + 1,
                        'Company_Name': 'Empty/Invalid Name',
                        'Company_Description': '',
                        'Found_Domain': 'Skipped',
                        'Search_Status': 'Skipped'
                    })
                
                # Save intermediate results every 20 companies
                if (index + 1) % 20 == 0:
                    self.save_results(results, f"intermediate_{output_path}")
                    logger.info(f"Intermediate results saved. Progress: {index + 1}/{total_companies}")
            
            # Save final results
            self.save_results(results, output_path)
            
            # Save domains-only file
            domains_only_path = output_path.replace('.csv', '_domains_only.txt')
            self.save_domains_only(results, domains_only_path)
            
            # Print summary
            logger.info(f"Scraping completed! Found domains for {found_count}/{total_companies} companies")
            logger.info(f"Success rate: {found_count/total_companies*100:.1f}%")
            logger.info(f"Full results saved to: {output_path}")
            logger.info(f"Domains only saved to: {domains_only_path}")
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            raise
    
    def save_results(self, results: List[dict], output_path: str):
        """Save results to CSV file."""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Row_Number', 'Company_Name', 'Company_Description', 'Found_Domain', 'Search_Status']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
                    
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def save_domains_only(self, results: List[dict], output_path: str):
        """Save only the domains to a text file in order."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Website Domains (in order from Excel file)\n")
                f.write("# Format: Row_Number. Company_Name -> Domain\n\n")
                
                for result in results:
                    domain = result['Found_Domain']
                    if domain and domain != 'Not Found' and domain != 'Skipped':
                        f.write(f"{result['Row_Number']}. {result['Company_Name']} -> {domain}\n")
                    else:
                        f.write(f"{result['Row_Number']}. {result['Company_Name']} -> {domain}\n")
                        
        except Exception as e:
            logger.error(f"Error saving domains only file: {str(e)}")
            raise

def main():
    """Main function to run the scraper."""
    excel_file = 'Growth For Impact Data Assignment.xlsx'
    output_file = 'scraped_domains.csv'
    
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        logger.error(f"Excel file '{excel_file}' not found!")
        print(f"Please make sure '{excel_file}' is in the current directory.")
        return
    
    print("=== Website Domain Scraper ===")
    print(f"Processing: {excel_file}")
    print(f"Output will be saved to: {output_file}")
    print("\nThis may take a while depending on the number of companies...")
    print("Check domain_scraper.log for detailed progress.\n")
    
    # Create scraper instance
    scraper = ComprehensiveWebsiteScraper(delay=1.5)  # 1.5 second delay between requests
    
    # Process the file
    try:
        scraper.process_excel_file(excel_file, output_file)
        print(f"\n✅ Scraping completed successfully!")
        print(f"📊 Check the output file: {output_file}")
        print(f"📋 Domains only file: {output_file.replace('.csv', '_domains_only.txt')}")
        print(f"📝 Detailed log: domain_scraper.log")
        
    except Exception as e:
        logger.error(f"Scraping failed: {str(e)}")
        print(f"❌ Scraping failed. Check domain_scraper.log for details.")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()