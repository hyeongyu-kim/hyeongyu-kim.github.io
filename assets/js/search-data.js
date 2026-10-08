// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "Publications",
          description: "Test-time adaptation, domain generalization, and medical imaging.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-code",
          title: "Code",
          description: "Research code and project repositories.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-research-notes",
          title: "Research notes",
          description: "Notes on test-time adaptation, matrix tiling, and the Relax-to-TIR compiler pipeline.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/notes/";
          },
        },{id: "nav-interactive-lab",
          title: "Interactive lab",
          description: "Interactive test-time adaptation and matrix-tiling experiments by Hyeongyu Kim.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/lab/";
          },
        },{id: "nav-now",
          title: "Now",
          description: "Current work, research interests, and reading by Hyeongyu Kim.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/now/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "Education, professional experience, research, and selected honors.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-our-paper-sdc-uda-has-been-accepted-at-cvpr-2023",
          title: 'Our paper SDC-UDA has been accepted at CVPR 2023!',
          description: "",
          section: "News",},{id: "news-our-paper-buffer-tta-has-been-accepted-at-neurips-2025",
          title: 'Our paper Buffer TTA has been accepted at NeurIPS 2025!',
          description: "",
          section: "News",},{id: "news-our-papers-actta-amp-amp-cd-buffer-have-been-accepted-at-cvpr-2026",
          title: 'Our papers AcTTA &amp;amp;amp; CD-Buffer have been accepted at CVPR 2026!',
          description: "",
          section: "News",},{id: "news-in-august-2026-i-joined-hyundai-motor-company-as-a-compiler-engineer-in-the-compiler-team-working-on-npu-compilation",
          title: 'In August 2026, I joined Hyundai Motor Company as a Compiler Engineer in...',
          description: "",
          section: "News",},{id: "news-i-received-my-ph-d-from-yonsei-university-advised-by-prof-dosik-hwang",
          title: 'I received my Ph.D. from Yonsei University, advised by Prof. Dosik Hwang.',
          description: "",
          section: "News",},{id: "news-our-seg-a-challenge-overview-is-available-online-in-medical-image-analysis",
          title: 'Our SEG.A challenge overview is available online in Medical Image Analysis.',
          description: "",
          section: "News",},{id: "news-mira-our-work-on-mutual-information-guided-calibration-for-test-time-adaptation-was-accepted-at-neurips-2026",
          title: 'MIRA, our work on mutual-information-guided calibration for test-time adaptation, was accepted at NeurIPS...',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/projects/9_project/";
            },},{id: "teachings-data-science-fundamentals",
          title: 'Data Science Fundamentals',
          description: "This course covers the foundational aspects of data science, including data collection, cleaning, analysis, and visualization. Students will learn practical skills for working with real-world datasets.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/data-science-fundamentals/";
            },},{id: "teachings-introduction-to-machine-learning",
          title: 'Introduction to Machine Learning',
          description: "This course provides an introduction to machine learning concepts, algorithms, and applications. Students will learn about supervised and unsupervised learning, model evaluation, and practical implementations.",
          section: "Teachings",handler: () => {
              window.location.href = "/teachings/introduction-to-machine-learning/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6B%68%67%34%33%30%39@%6E%61%76%65%72.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=Ot6fq-EAAAAJ", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/hyeongyu-kim", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/hyeongyu-kim-27b01b289", "_blank");
        },
      },{
        id: 'social-cv',
        title: 'CV',
        section: 'Socials',
        handler: () => {
          window.open("/assets/pdf/Hyeongyu_Kim_CV.pdf", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
