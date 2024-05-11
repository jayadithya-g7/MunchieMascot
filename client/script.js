import bot from "./assets/chef-hat-chef-svgrepo-com.svg";
import user from "./assets/user.svg";
import Cookies from "js-cookie";

const responses = {
  "who are you": [
    "I am a culinary AI companion!",
    "I'm your friendly kitchen assistant powered by artificial intelligence.",
    "I'm your personal cooking buddy here to help you in the kitchen!",
    "I'm MunchieBot, your digital sous chef ready to assist you with all things culinary!",
  ],
  "what can you do": [
    "I can suggest recipes, provide cooking tips, offer ingredient substitutions, and much more!",
    "I can help you find recipes, modify them to fit your dietary needs, and even provide cooking techniques!",
    "I can assist you with recipe suggestions, nutritional information, and answer any cooking-related queries you have!",
    "I'm here to help you with all things cooking! Just ask me for recipe ideas, cooking techniques, or ingredient substitutions.",
  ],
};

function getRandomResponse(query) {
  const responsesForQuery = responses[query];
  if (responsesForQuery) {
    const randomIndex = Math.floor(Math.random() * responsesForQuery.length);
    return responsesForQuery[randomIndex];
  } else return undefined;
}

if (!Cookies.get("userhash")) window.location.pathname = "/signin/";

const form = document.querySelector("form");
const chatContainer = document.querySelector("#chat_container");
const promptInput = document.querySelector('textarea[name="prompt"]');
const formSumbitButton = document.querySelector("#form-button");
const signoutButton = document.querySelector("#signout-button");
const gettingStartedContainer = document.querySelector(
  "#getting_started_container"
);

let loadInterval;

signoutButton.addEventListener("click", () => {
  Cookies.remove("userhash");
  window.location.reload();
});

function handleOnClickRecipeRef() {
  document.querySelectorAll('li[data-hasreciperef="true"').forEach((e) => {
    e.addEventListener("click", () => {
      const name = e.dataset.recipename;
      promptInput.value = `get recipe <span class='font-semibold'>${name}</span><span class='hidden' id="recipeid">${e.dataset.recipeid}</span>`;
      formSumbitButton.click();
    });
  });
}

function loader(element) {
  element.textContent = "";

  loadInterval = setInterval(() => {
    // Update the text content of the loading indicator
    element.textContent += ".";

    // If the loading indicator has reached three dots, reset it
    if (element.textContent === "....") {
      element.textContent = "";
    }
  }, 300);
}

function typeText(element, text) {
  let index = 0;
  element.innerHTML = text;
  handleOnClickRecipeRef();
  // let interval = setInterval(() => {
  //   if (index < text.length) {
  //     element.innerHTML += text.charAt(index);
  //     index++;
  //   } else {
  //     clearInterval(interval);
  //   }
  // }, 20);
}

// generate unique ID for each message div of bot
// necessary for typing text effect for that specific reply
// without unique ID, typing text will work on every element
function generateUniqueId() {
  const timestamp = Date.now();
  const randomNumber = Math.random();
  const hexadecimalString = randomNumber.toString(16);

  return `id-${timestamp}-${hexadecimalString}`;
}

function chatStripe(isAi, value, uniqueId) {
  return `
        <div class="wrapper ${isAi && "ai"}">
            <div class="chat">
                <div class="profile">
                    <img 
                      src=${isAi ? bot : user} 
                      alt="${isAi ? "bot" : "user"}" 
                    />
                </div>
                <div class="message" id=${uniqueId}>${value}</div>
            </div>
        </div>
    `;
}

const handleSubmit = async (e) => {
  e.preventDefault();

  const data = new FormData(form);
  const prompt = data.get("prompt").trim().toLowerCase();
  // user's chatstripe
  chatContainer.innerHTML += chatStripe(false, prompt);
  if (chatContainer.innerHTML.length > 0) {
    chatContainer.classList.replace("hidden", "flex");
    gettingStartedContainer.classList.add("hidden");
  }
  // to clear the textarea input
  form.reset();

  // bot's chatstripe
  const uniqueId = generateUniqueId();
  chatContainer.innerHTML += chatStripe(true, " ", uniqueId);

  // to focus scroll to the bottom
  chatContainer.scrollTop = chatContainer.scrollHeight;

  // specific message div
  const messageDiv = document.getElementById(uniqueId);

  // messageDiv.innerHTML = "..."
  let response;
  let promptType = 0;
  loader(messageDiv);
  console.log(prompt);
  const rr = getRandomResponse(prompt);
  if (rr) {
    promptType = -1;
    response = rr;
  } else if (prompt.indexOf("get recipe") === 0) {
    promptType = 1;
    var parser = new DOMParser();
    var doc = parser.parseFromString(prompt, "text/html");
    var span = doc.querySelector("#recipeid");
    response = await fetch(
      `http://127.0.0.1:5000/recipe/${span.innerText}?` +
        new URLSearchParams({
          userhash: Cookies.get("userhash"),
        })
    );
  } else {
    response = await fetch(
      "http://127.0.0.1:5000/recipe/search?" +
        new URLSearchParams({
          userhash: Cookies.get("userhash"),
          q: prompt,
        })
    );
  }
  clearInterval(loadInterval);
  messageDiv.innerHTML = " ";

  if (promptType !== -1 && response.ok) {
    const data = await response.json();
    let content;
    switch (promptType) {
      case 0:
        if (data.length === 0) {
          content = "Sorry, I'm not sure how to respond to that.";
        }
        content = "Here are some results...\n";
        data.forEach((v) => {
          content += `<li class="capitalize" data-hasreciperef="true" data-recipeid="${v.id}" data-recipename="${v.name}">${v.name}</li>`;
        });
        break;
      case 1:
        content = `<span class="capitalize font-semibold text-blue-600">${data.name}</span><p class="capitalize font-semibold">Ingredients</p><ul class="flex items-center text-sm flex-wrap gap-2">`;
        data.ingredients.forEach((v) => {
          content += `<li class="whitespace-nowrap bg-black/20 rounded-full px-2 capitalize">${v}</li>`;
        });
        content += `</ul><p class="capitalize font-semibold">Steps</p><ul class="flex flex-col list-inside list-disc">`;
        data.steps.forEach((v) => {
          content += `<li class="px-2 capitalize">${v}</li>`;
        });
        content += "</ul>";
      default:
        break;
    }
    typeText(messageDiv, content);
  } else if (promptType === -1) {
    let content;
    content = `<span class="capitalize">${response}</span>`;
    typeText(messageDiv, content);
  } else {
    const err = await response.text();

    messageDiv.innerHTML = "Something went wrong";
    alert(err);
  }
};

form.addEventListener("submit", handleSubmit);
