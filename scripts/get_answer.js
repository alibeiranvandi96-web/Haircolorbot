const fs = require('fs');

// خواندن دیتا از فایل JSON
const data = JSON.parse(fs.readFileSync('data/hair_data.json', 'utf8'));

// سوال کاربر از آرگومان خط فرمان
const userQuestion = (process.argv[2] || '').toLowerCase();

// حذف علائم و اعداد اضافی برای پردازش بهتر
const cleanQuestion = userQuestion.replace(/[?؟!.,،]/g, ' ');

let bestAnswer = "متاسفانه در دیتای من پاسخی برای این سوال پیدا نکردم. لطفا سوال خود را دقیق‌تر بپرسید. می‌توانید درباره: تناژهای رنگ (طبیعی، دودی، مسی و...)، پایه‌های دکلره، فرمول نویسی ترکیب رنگ، واریاسیون‌ها یا رنگ‌های ثابت بپرسید.";
let maxScore = 0;

for (const item of data.questions) {
  let score = 0;
  
  // بررسی کلمات کلیدی
  for (const keyword of item.keywords) {
    if (cleanQuestion.includes(keyword.toLowerCase())) {
      score += 3; // امتیاز بیشتر برای تطابق کامل کلمه
    }
  }
  
  // اگر کلمه کلیدی در سوال بود، امتیاز بده
  if (score > maxScore) {
    maxScore = score;
    bestAnswer = item.answer;
  }
}

console.log(bestAnswer);
