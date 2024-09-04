// Full questions_dict and additional_questions_dict
const questionsDict = {
    'Q1A': "I found myself getting upset by quite trivial things.",
    'Q2A': "I was aware of dryness of my mouth.",
    'Q3A': "I couldn't seem to experience any positive feeling at all.",
    'Q4A': "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness in the absence of physical exertion).",
    'Q5A': "I just couldn't seem to get going.",
    'Q6A': "I tended to over-react to situations.",
    'Q7A': "I had a feeling of shakiness (e.g., legs going to give way).",
    'Q8A': "I found it difficult to relax.",
    'Q9A': "I found myself in situations that made me so anxious I was most relieved when they ended.",
    'Q10A': "I felt that I had nothing to look forward to.",
    'Q11A': "I found myself getting upset rather easily.",
    'Q12A': "I felt that I was using a lot of nervous energy.",
    'Q13A': "I felt sad and depressed.",
    'Q14A': "I found myself getting impatient when I was delayed in any way (e.g., elevators, traffic lights, being kept waiting).",
    'Q15A': "I had a feeling of faintness.",
    'Q16A': "I felt that I had lost interest in just about everything.",
    'Q17A': "I felt I wasn't worth much as a person.",
    'Q18A': "I felt that I was rather touchy.",
    'Q19A': "I perspired noticeably (e.g., hands sweaty) in the absence of high temperatures or physical exertion.",
    'Q20A': "I felt scared without any good reason.",
    'Q21A': "I felt that life wasn't worthwhile.",
    'Q22A': "I found it hard to wind down.",
    'Q23A': "I had difficulty in swallowing.",
    'Q24A': "I couldn't seem to get any enjoyment out of the things I did.",
    'Q25A': "I was aware of the action of my heart in the absence of physical exertion (e.g., sense of heart rate increase, heart missing a beat).",
    'Q26A': "I felt down-hearted and blue.",
    'Q27A': "I found that I was very irritable.",
    'Q28A': "I felt I was close to panic.",
    'Q29A': "I found it hard to calm down after something upset me.",
    'Q30A': "I feared that I would be 'thrown' by some trivial but unfamiliar task.",
    'Q31A': "I was unable to become enthusiastic about anything.",
    'Q32A': "I found it difficult to tolerate interruptions to what I was doing.",
    'Q33A': "I was in a state of nervous tension.",
    'Q34A': "I felt I was pretty worthless.",
    'Q35A': "I was intolerant of anything that kept me from getting on with what I was doing.",
    'Q36A': "I felt terrified.",
    'Q37A': "I could see nothing in the future to be hopeful about.",
    'Q38A': "I felt that life was meaningless.",
    'Q39A': "I found myself getting agitated.",
    'Q40A': "I was worried about situations in which I might panic and make a fool of myself.",
    'Q41A': "I experienced trembling (e.g., in the hands).",
    'Q42A': "I found it difficult to work up the initiative to do things."
  };
  
  const additionalQuestionsDict = {
    'TIPI1': "Extraverted, enthusiastic.",
    'TIPI2': "Critical, quarrelsome.",
    'TIPI3': "Dependable, self-disciplined.",
    'TIPI4': "Anxious, easily upset.",
    'TIPI5': "Open to new experiences, complex.",
    'TIPI6': "Reserved, quiet.",
    'TIPI7': "Sympathetic, warm.",
    'TIPI8': "Disorganized, careless.",
    'TIPI9': "Calm, emotionally stable.",
    'TIPI10': "Conventional, uncreative."
  };
  
  const demographicsDict = {
    'education': "How much education have you completed? (1=Less than high school, 2=High school, 3=University degree, 4=Graduate degree)",
    'urban': "What type of area did you live when you were a child? (1=Rural, 2=Suburban, 3=Urban)",
    'gender': "What is your gender? (1=Male, 2=Female, 3=Other)",
    'religion': "What is your religion? (1=Agnostic, 2=Atheist, 3=Buddhist, 4=Christian (Catholic), 5=Christian (Mormon), 6=Christian (Protestant), 7=Christian (Other), 8=Hindu, 9=Jewish, 10=Muslim, 11=Sikh, 12=Other)",
    'race': "What is your race? (10=Asian, 20=Arab, 30=Black, 40=Indigenous Australian, 50=Native American, 60=White, 70=Other)",
    'married': "What is your marital status? (1=Never married, 2=Currently married, 3=Previously married)",
    'familysize': "Including you, how many children did your mother have?",
    'age_group': "Age group of the respondent (derived from their actual age)."
  };
  
  const questionsArray = Object.entries({ ...questionsDict, ...additionalQuestionsDict, ...demographicsDict });  // Combine all questions
  
  const totalQuestions = questionsArray.length;
  let currentQuestionIndex = 0;
  
  // Populate the first question initially
  function populateQuestions() {
    const questionsContainer = document.getElementById('questionsContainer');
    
    questionsArray.forEach(([key, question], index) => {
      const questionDiv = document.createElement('div');
      questionDiv.classList.add('question');
      if (index === 0) questionDiv.classList.add('active'); // Show first question
  
      const label = document.createElement('label');
      label.textContent = question;
  
      const select = document.createElement('select');
      select.name = key;
  
      // Determine options based on question type
      if (key.startsWith('TIPI') || Object.keys(demographicsDict).includes(key)) {
        // TIPI and demographics questions
        for (let i = 1; i <= 7; i++) {
          const option = document.createElement('option');
          option.value = i;
          option.textContent = i;
          select.appendChild(option);
        }
      } else {
        // Regular questions
        for (let i = 1; i <= 4; i++) {
          const option = document.createElement('option');
          option.value = i;
          option.textContent = i;
          select.appendChild(option);
        }
      }
  
      questionDiv.appendChild(label);
      questionDiv.appendChild(select);
      questionsContainer.appendChild(questionDiv);
    });
  }
  
  populateQuestions();
  
  // Next button handler
  function nextQuestion() {
    const questions = document.querySelectorAll('.question');
    if (currentQuestionIndex < totalQuestions - 1) {
      questions[currentQuestionIndex].classList.remove('active');
      currentQuestionIndex++;
      questions[currentQuestionIndex].classList.add('active');
      document.getElementById('prevBtn').disabled = false;
    }
  
    if (currentQuestionIndex === totalQuestions - 1) {
      document.getElementById('nextBtn').textContent = 'Submit';
      document.getElementById('nextBtn').onclick = submitForm;
    }
  
    updateProgress();
  }
  
  // Previous button handler
  function prevQuestion() {
    const questions = document.querySelectorAll('.question');
    if (currentQuestionIndex > 0) {
      questions[currentQuestionIndex].classList.remove('active');
      currentQuestionIndex--;
      questions[currentQuestionIndex].classList.add('active');
      document.getElementById('nextBtn').textContent = 'Next';
      document.getElementById('nextBtn').onclick = nextQuestion;
    }
  
    if (currentQuestionIndex === 0) {
      document.getElementById('prevBtn').disabled = true;
    }
  
    updateProgress();
  }
  
  // Update progress bar
  function updateProgress() {
    const progress = document.getElementById('progress');
    progress.style.width = ((currentQuestionIndex + 1) / totalQuestions) * 100 + '%';
  }
  
  // Submit the form
  function submitForm() {
    const form = document.getElementById('surveyForm');
    const formData = new FormData(form);
  
    const rowData = [];
    formData.forEach((value, key) => {
      rowData.push(value);
    });
  
    // Convert form data into CSV format using PapaParse
    const csv = Papa.unparse([rowData]);
  
    // Download the CSV file
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', 'survey_results.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
  