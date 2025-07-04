# Legal Chat Frontend

## 디렉토리 구조

	/seasonal-ai-law-chatbot-frontend
    └───legal_chat/
	    ├───.editorconfig
	    ├───.env.development
	    ├───.env.production
	    ├───.gitattributes
	    ├───.gitignore
	    ├───.prettierrc.json
	    ├───eslint.config.js
	    ├───index.html
	    ├───jsconfig.json
	    ├───package-lock.json
	    ├───package.json
	    ├───README.md
	    ├───vite.config.js
	    ├───.vscode/
	    │   └───extensions.json
	    └───src/
	        ├───App.vue
	        ├───main.js
	        ├───assets/
	        │   └───legal_bot.png
	        ├───components/
	        │   ├───ChatBot.vue
	        │   ├───Footer.vue
	        │   └───Header.vue
	        ├───router/
	        │   └───index.js
	        ├───stores/
	        │   └───counter.js
	        └───views/
	            ├───ChatPage.vue
	            └───MainPage.vue

## 주요 파일 및 디렉토리 설명

### Root Directory

- **`.editorconfig`**: 다양한 편집기와 IDE에서 코드 스타일을 일관되게 유지하기 위한 설정 파일입니다.
- **`.env.development`**: 개발 환경에서 사용되는 환경 변수를 정의하는 파일입니다.
- **`.env.production`**: 프로덕션 환경에서 사용되는 환경 변수를 정의하는 파일입니다.
- **`.gitignore`**: Git 버전 관리에서 제외할 파일 및 디렉토리 목록을 지정합니다.
- **`.prettierrc.json`**: 코드 포맷터인 Prettier의 설정 파일입니다.
- **`eslint.config.js`**: JavaScript/Vue 코드의 문법 오류나 스타일을 검사하는 ESLint의 설정 파일입니다.
- **`index.html`**: Vue 애플리케이션이 마운트되는 기본 HTML 파일입니다.
- **`jsconfig.json`**: JavaScript 프로젝트의 설정 파일로, 주로 절대 경로 별칭 등을 설정하는 데 사용됩니다.
- **`package.json`**: 프로젝트의 정보, 의존성, 실행 스크립트 등을 정의하는 파일입니다.
- **`package-lock.json`**: 의존성 패키지들의 정확한 버전 정보를 저장하여 일관된 설치를 보장합니다.
- **`vite.config.js`**: 빠르고 효율적인 웹 개발 빌드 도구인 Vite의 설정 파일입니다.

### `src` Directory

- **`src/`**: 애플리케이션의 핵심 소스 코드가 위치하는 디렉토리입니다.
- **`src/main.js`**: Vue 인스턴스를 생성하고 애플리케이션을 초기화하는 메인 JavaScript 파일입니다.
- **`src/App.vue`**: 애플리케이션의 최상위 루트 컴포넌트입니다.
- **`src/assets/`**: 이미지, 폰트, 스타일시트 등 정적 자산을 저장하는 디렉토리입니다.
- **`src/components/`**: 애플리케이션 전반에서 재사용되는 컴포넌트(예: `Header`, `Footer`)를 모아두는 디렉토리입니다.
- **`src/router/`**: Vue Router를 사용하여 페이지 라우팅을 관리하는 설정 파일이 위치합니다.
- **`src/stores/`**: Pinia와 같은 상태 관리 라이브러리를 사용하여 애플리케이션의 전역 상태를 관리하는 파일이 위치합니다.
- **`src/views/`**: 각 라우트에 해당하는 페이지 레벨의 컴포넌트(예: `MainPage`, `ChatPage`)를 저장하는 디렉토리입니다.

## 🧾 SSAFY Legal ChatBot - Frontend

**법률에 대한 질문을 빠르게 해결해주는 챗봇 프론트엔드입니다.**  
Vue 3 + Vite + TailwindCSS 기반으로 개발되었습니다.

---

### 📸 주요 화면

- 홈 화면:  
  “SSAFY Legal Chatbot” 소개 및 시작하기 버튼  
  

- 챗봇 화면:  
  사용자의 법률 질문과 챗봇 응답 대화 UI  
  

---

### ⚙️ 실행 방법

#### 1. 의존성 설치

```
npm install
```
#### 2. .env 파일 설정
.env 또는 .env.local 파일을 프로젝트 루트에 생성한 뒤, 아래와 같이 API 주소를 명시하세요:
```
VITE_API_ENDPOINT=https://aicourse-backend.duckdns.org
VITE_FALLBACK_API_ENDPOINT=https://seasonal-ai-law-chatbot-backend.fly.dev
VITE_USE_MOCK=false  
```
✅ .env 파일이 누락되면 챗봇 API 호출이 작동하지 않습니다.

#### 3. 개발 서버 실행
```
npm run dev
```
실행 후, 브라우저에서 http://localhost:5173 주소로 접속합니다.

---

### 🧪 빌드
```
npm run build
```
빌드 결과는 dist/ 폴더에 생성되며, EC2 등의 서버에 배포할 수 있습니다.

---

### 📝 기타 사항
TailwindCSS가 적용되어 있어, 커스터마이징이 쉽고 반응형 UI가 기본 적용되어 있습니다.

챗봇 API와 통신 시 CORS 문제가 발생하면 백엔드 서버에 CORS 허용 설정이 필요합니다.

"내용 초기화" 버튼은 로컬 IndexedDB의 대화 저장 내용을 초기화합니다.