- Front-end URL
`https://main.d1wu48zue8p0tl.amplifyapp.com/chat`

## 프로젝트 구조

```
legal_chat/
├───src/
│   ├───assets/
│   ├───components/
│   ├───router/
│   ├───stores/
│   └───views/
├───index.html
├───package.json
└───vite.config.js
```

## `src` 프로젝트 구조

*   **`main.js`**: 애플리케이션의 진입점입
*   **`App.vue`**: 애플리케이션의 루트 컴포넌트
*   **`components/`**: 재사용 가능한 Vue 컴포넌트
    *   `ChatBot.vue`: 챗봇 UI의 핵심 컴포넌트
    *   `Header.vue`: 페이지 상단의 헤더 컴포넌트
    *   `Footer.vue`: 페이지 하단의 푸터 컴포넌트
*   **`router/`**: Vue Router 설정을 관리하는 디렉토리
*   **`stores/`**: Pinia 또는 Vuex와 같은 상태 관리 라이브러리의 스토어 저장
*   **`views/`**: 특정 라우트에 매핑되는 페이지 레벨의 컴포넌트를 저장
    *   `MainPage.vue`: 메인 페이지 컴포넌트
    *   `ChatPage.vue`: 챗봇과 대화하는 채팅 페이지 컴포넌트