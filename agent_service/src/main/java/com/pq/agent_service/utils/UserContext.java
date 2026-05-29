package com.pq.agent_service.utils;

import lombok.Data;

public class UserContext {

    private static final ThreadLocal<UserInfo> USER_THREAD_LOCAL = new ThreadLocal<>();

    @Data
    public static class UserInfo {
        private Long id;
        private String username;
    }

    public static void setUser(Long id, String username) {
        UserInfo info = new UserInfo();
        info.setId(id);
        info.setUsername(username);
        USER_THREAD_LOCAL.set(info);
    }

    public static UserInfo getUser() {
        return USER_THREAD_LOCAL.get();
    }

    public static void removeUser() {
        USER_THREAD_LOCAL.remove();
    }
}
