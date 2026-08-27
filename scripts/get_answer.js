const fs = require('fs');

// خواندن دیتا از فایل JSON
const data = JSON.parse(fs.readFileSync('data/hair_data.json', 'utf8'));

// سوال کاربر (از آرگومان خط فرمان میاد)
const userQuestion = process.argv[2] || '';
const lowerQuestion = userQuestion.toLowerCase();

// جستجو برای پیدا کردن بهترین پاسخ
let answer = "متاسفانه در دیتای من پاسخی برای این سوال پیدا نکردم. لطفا سوال خود را دقیق‌تر بپرسید. می‌توانید درباره: تناژهای رنگ (طبیعی، دودی، مسی و...)، پایه‌های دکلره، فرمول نویسی ترکیب رنگ، واریاسیون‌ها یا رنگ‌های ثابت (شرابی، ماهگونی و...) بپرسید.";

let bestMatchCount = 0;
let bestAnswer = "";

for (const item of data.questions) {
  // بررسی می‌کنیم آیا کلمات کلیدی در سوال کاربر هست یا نه
  const matchCount = item.keywords.filter(keyword => lowerQuestion.includes(keyword)).length;
  
  if (matchCount > bestMatchCount) {
    bestMatchCount = matchCount;
    bestAnswer = item.answer;
  }
}

if (bestMatchCount > 0) {
  answer = bestAnswer;
}

// چاپ پاسخ برای استفاده در GitHub Actions
console.log(answer);
