import httpx
import time
import json
import xml.etree.ElementTree as ET
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from typing import List
from document import Document
from loader import Loader


class SitemapLoader(Loader):
    url: str

    def fetch_sitemap_urls(self, sitemap_url):
        if not sitemap_url.endswith('.xml'):
            sitemap_url += '/sitemap.xml'  # Append '/sitemap.xml' if not present

        try:
            with httpx.Client() as client:
                response = client.get(sitemap_url)
                response.raise_for_status()  # Raise an error for bad responses

                # Parse the XML content
                root = ET.fromstring(response.content)

                urls = [elem.text for elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]

                # If no URLs are found, try without the namespace
                if not urls:
                    urls = [elem.text for elem in root.findall('.//loc')]

                if not urls:
                    raise ValueError("Could not parse sitemap format")

                print(f"Found {len(urls)} URLs in sitemap: {sitemap_url}.")
                return urls

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            raise RuntimeError(f"Failed to fetch a sitemap at {sitemap_url}") from e

    def initialize_driver(self, headless=False):
        options = Options()
        if headless:
            options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        return driver

    def extract_text_from_soup(self, soup):
        text_parts = []

        def extract_text_from_tag(tag):
            # Extract standard text content
            text_content = tag.get_text(separator=' ', strip=True)
            if text_content:
                text_parts.append(text_content)

            # Extract additional text from specific attributes
            for attr in ['data-sheets-value', 'alt', 'title']:
                attr_content = tag.get(attr)
                if attr_content:
                    try:
                        # Attempt to parse JSON content
                        json_content = json.loads(attr_content)
                        if isinstance(json_content, dict) and '1' in json_content and isinstance(json_content['1'], dict) and '2' in json_content['1']:
                            text_parts.append(json_content['1']['2'])
                        else:
                            # Handle non-JSON content or JSON without the expected structure
                            text_parts.append(str(json_content))
                    except json.JSONDecodeError:
                        # Handle non-JSON content
                        text_parts.append(attr_content)

        # Define tags to include in text extraction
        tags_of_interest = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'a', 'div']
        for tag in soup.find_all(tags_of_interest):
            extract_text_from_tag(tag)

        # Join extracted parts with newline characters
        extracted_text = '\n'.join(text_parts)
        return extracted_text

    def clean_soup(self, soup):
        # Specify the tags you want to remove
        tags_to_remove = ["script", "style", "head", "meta", "svg", "iframe", "nav", "footer", "form", "noscript", "header"]
        for tag in tags_to_remove:
            for s in soup.find_all(tag):
                s.extract()
        return soup

    def load_url(self, driver, url, timeout=10, post_load_sleep=0):
        driver.get(url)
        try:
            WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "title")))
            page_title = driver.title
        except TimeoutException:
            print(f"Timed out waiting for the title tag for: {url}")
            return None, None
        time.sleep(post_load_sleep)  # Additional wait for dynamic content
        return driver.page_source, page_title

    def process_url(self, driver, url, post_load_sleep=0):
        try:
            page_content, page_title = self.load_url(driver, url, post_load_sleep=post_load_sleep)
            if page_content is None:
                return None
            else:
                soup = BeautifulSoup(page_content, 'html.parser')
                soup = self.clean_soup(soup)  # Clean the soup to remove unnecessary tags
                text = self.extract_text_from_soup(soup)
                doc = Document(content=text, title=page_title, source=url)
                return doc
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            return None

    def load_urls_with_sleep(self, urls, headless=False, post_load_sleep=5):
        documents = []
        with self.initialize_driver(headless=headless) as driver:
            for url in urls:
                doc = self.process_url(driver, url, post_load_sleep=post_load_sleep)
                if doc:
                    documents.append(doc)
        return documents

    def load_documents(self) -> List[Document]:
        sitemap_urls = self.fetch_sitemap_urls(sitemap_url=self.url)
        return self.load_urls_with_sleep(sitemap_urls)
