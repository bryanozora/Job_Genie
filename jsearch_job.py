import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

# --- CONFIG ---
API_KEY = "rapid_api_key"  # <<<<<<<<<<< paste your key here
TOTAL_PAGES_PER_CATEGORY = 4  # <-- how many pages PER category to scrape
CATEGORIES = CATEGORIES = CATEGORIES = [
    "accountant", "advocate", "agriculture", "apparel", "arts",
    "automobile", "aviation", "banking", "customer support", "business development",
    "chef", "construction", "consultant", "designer", "digital marketing",
    "engineering", "finance", "fitness", "healthcare", "hr",
    "information technology", "public relations", "sales", "teacher"
]


def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator="\n").strip()

def scrape_and_save_jobs(filename="jobs.csv"):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    jobs = []

    for category in CATEGORIES:
        for page in range(1, TOTAL_PAGES_PER_CATEGORY + 1):
            querystring = {"query": category, "remote_jobs_only": "true", "page": str(page)}

            response = requests.get(url, headers=headers, params=querystring)

            if response.status_code == 200:
                data = response.json().get("data", [])
                for job in data:
                    jobs.append({
                        "Job Title": job.get("job_title", ""),
                        "Company": job.get("employer_name", ""),
                        "Location": job.get("job_city") or "Remote",
                        "Salary": f"{job.get('job_min_salary', '')} - {job.get('job_max_salary', '')} {job.get('job_salary_currency', '')}" if job.get("job_min_salary") else "Not specified",
                        "Job Description": clean_html(job.get("job_description", "")),
                        "Job Link": job.get("job_apply_link", ""),
                        "Category": category  # <<--- keep track of which category it came from
                    })
                print(f"✅ Scraped {len(data)} jobs for category '{category}' page {page}")
            else:
                print(f"❌ Failed to fetch {category} page {page}. Status: {response.status_code}")

            time.sleep(1)  # Be nice to the API, short sleep between pages

    df = pd.DataFrame(jobs)
    df.to_csv(filename, index=False)
    print(f"✅ Total {len(jobs)} jobs saved to {filename}")

if __name__ == "__main__":
    scrape_and_save_jobs()
