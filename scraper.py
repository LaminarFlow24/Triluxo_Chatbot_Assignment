import time
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# --- Selenium Setup (Headless Chrome) ---
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run headless
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=chrome_options)

# URL of the website to scrape (main courses page)
main_url = "https://brainlox.com/courses/category/technical"  # Replace if needed

# Get the main courses page using requests and BeautifulSoup
response = requests.get(main_url)
soup = BeautifulSoup(response.text, "html.parser")

courses = []

# Find all course containers on the main page
div_containers = soup.find_all("div", class_="col-lg-6 col-md-12")

for container in div_containers:
    course = {}
    
    # Extract Course Name from the main page
    course_name_tag = container.find("h3").find("a")
    course["Course Name"] = course_name_tag.text.strip()
    
    # Extract Course Link and form full URL for the course details page
    course_link = course_name_tag["href"]
    full_course_link = f"https://brainlox.com/{course_link}"
    course["Course Link"] = full_course_link
    
    # Extract Course Details (short description)
    course_details_tag = container.find("p")
    course["Course Details"] = course_details_tag.text.replace("DESCRIPTION", "").strip()
    
    # Extract Course Price
    price_tag = container.find("div", class_="price")
    if price_tag:
        course["Course Price"] = price_tag.find("span", class_="price-per-session").text.strip()
    else:
        course["Course Price"] = "Free or Not Listed"
    
    # Extract Number of Lessons
    lessons_tag = container.find("ul", class_="courses-box-footer").find("li")
    course["Number of Lessons"] = lessons_tag.text.strip().split()[0]  # Get the number part
    
    # --- Use Selenium to load the full course page ---
    driver.get(full_course_link)
    time.sleep(3)  # Wait for the page to fully load
    
    # --- Step 1: Get Course Description BEFORE clicking on "Curriculum" ---
    page_source = driver.page_source
    course_soup = BeautifulSoup(page_source, "html.parser")
    description_tag = course_soup.find("div", class_="courses-overview")
    if description_tag:
        p_tag = description_tag.find("p")
        course["Course Description"] = p_tag.text.strip() if p_tag else "No description available"
    else:
        course["Course Description"] = "No description available"
    
    # --- Step 2: Click the "Curriculum" tab to load curriculum content ---
    try:
        # Find all the react-tabs tab elements and click the one with "Curriculum"
        tabs = driver.find_elements(By.CSS_SELECTOR, "li.react-tabs__tab")
        for tab in tabs:
            if "Curriculum" in tab.text:
                # If not already selected, click it.
                if tab.get_attribute("aria-selected") != "true":
                    tab.click()
                    time.sleep(2)  # Wait for curriculum content to load
                break
    except Exception as e:
        print(f"Error clicking Curriculum tab on {full_course_link}: {e}")
    
    # --- Step 3: After clicking, get the updated page source and extract curriculum ---
    updated_page_source = driver.page_source
    updated_soup = BeautifulSoup(updated_page_source, "html.parser")
    
    # Extract all <span class="courses-name"> elements
    curriculum_spans = updated_soup.find_all("span", class_="courses-name")
    curriculum_texts = [span.text.strip() for span in curriculum_spans if span.text.strip()]
    
    # Store the curriculum text in the course dictionary
    if curriculum_texts:
        course["Course Curriculum"] = curriculum_texts
    else:
        course["Course Curriculum"] = "No curriculum available"
    
    # Add the course dictionary to the courses list
    courses.append(course)

# Close the Selenium driver
driver.quit()

# Save the courses data to a JSON file
with open("courses.json", "w", encoding="utf-8") as f:
    json.dump(courses, f, indent=4)

# Print the extracted data
print(json.dumps(courses, indent=4))
